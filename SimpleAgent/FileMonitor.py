import asyncio
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import logging
from dotenv import load_dotenv
import hashlib
import queue


class NoToolNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg.startswith("Warning: there are non-text parts in the response: ['function_call']"):
            return False
        return True

# Load environment variables
load_dotenv()

# Constants
APP_NAME = "document_summarizer"
GEMINI_2_FLASH = "gemini-2.0-flash-lite"
WATCH_DIRECTORY = "./watched_files"  # İzlenecek klasör

class DocumentSummarizerAgent:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.processed_files = set()  # İşlenmiş dosyaları takip etmek için
        self.file_queue = queue.Queue()  # Dosya işleme kuyruğu
        self.loop = None  # Event loop referansı
        
        # Özetleme ajanı
        self.summarizer_agent = LlmAgent(
            name="document_summarizer",
            model=GEMINI_2_FLASH,
            instruction="""
            Sen bir belge özetleme uzmanısın. Kullanıcı sana bir belge içeriği verecek ve sen bu belgeyi analiz edeceksin.
            
            Görevin:
            1. Verilen belge içeriğini dikkatli bir şekilde oku ve analiz et
            2. Metnin ana konusunu ve temasını belirle
            3. Önemli noktaları, anahtar bilgileri ve ana argümanları çıkar
            4. Varsa sayısal veriler, tarihler, isimler ve önemli detayları dahil et
            5. Kapsamlı ama öz bir özet oluştur (3-5 paragraf)
            6. Özeti Türkçe olarak sun
            
            Yanıtını şu formatta ver:
            📄 BELGE ÖZETİ
            🔸 Ana Konu: [ana konu ve tema]
            🔸 Anahtar Noktalar: [önemli noktalar listesi]
            🔸 Detaylı Özet: [kapsamlı özet]
            🔸 Önemli Detaylar: [sayılar, tarihler, isimler vs.]
            """,
            tools=[],
        )
        
        # Dizinleri oluştur
        os.makedirs(WATCH_DIRECTORY, exist_ok=True)
        
        print(f"📁 İzlenen klasör: {os.path.abspath(WATCH_DIRECTORY)}")

    def set_event_loop(self, loop):
        """Event loop'u ayarla"""
        self.loop = loop

    def add_file_to_queue(self, file_path):
        """Dosyayı işleme kuyruğuna ekle"""
        self.file_queue.put(file_path)
        print(f"📥 Dosya kuyruğa eklendi: {os.path.basename(file_path)}")

    async def process_file_queue(self):
        """Dosya kuyruğunu sürekli işle"""
        while True:
            try:
                # Kuyruktan dosya al (non-blocking)
                try:
                    file_path = self.file_queue.get_nowait()
                    await self.process_new_file(file_path)
                    self.file_queue.task_done()
                except queue.Empty:
                    # Kuyruk boşsa kısa bekle
                    await asyncio.sleep(0.1)
            except Exception as e:
                print (f"Kuyruk işleme hatası: {str(e)}")
                await asyncio.sleep(1)

    def get_file_hash(self, file_path):
        """Dosyanın hash değerini hesapla (aynı dosyanın tekrar işlenmesini önlemek için)"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    async def summarize_document(self, file_path):
        """Belgeyi özetle"""
        try:
            # Dosya içeriğini oku
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print (f"Dosya boş: {file_path}")
                return None
            
            # Dosya hash'i kontrol et
            file_hash = self.get_file_hash(file_path)
            if file_hash in self.processed_files:
                print (f"Dosya zaten işlenmiş: {file_path}")
                return None
            
            # Session oluştur
            user_id = f"user_{int(time.time())}"
            session_id = f"session_{file_hash[:8]}"
            
            runner = Runner(
                agent=self.summarizer_agent,
                app_name=APP_NAME,
                session_service=self.session_service,
            )
            
            initial_state = {"document_content": content}
            
            await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state=initial_state
            )
            
            # Dosya içeriğini LLM'e gönder - tam içerik
            if len(content) > 10000:  # Çok uzun metinler için uyarı
                prompt = f"""
Aşağıdaki belgeyi analiz et ve özetle. Bu belge uzun olduğu için dikkatli oku:

BELGE İÇERİĞİ:
{content}

Bu belgeyi tamamen oku ve kapsamlı bir özet çıkar.
"""
            else:
                prompt = f"""
Aşağıdaki belgeyi analiz et ve özetle:

BELGE İÇERİĞİ:
{content}

Bu belgenin tüm önemli noktalarını kapsayan bir özet oluştur.
"""
            
            user_content = types.Content(
                role='user', 
                parts=[types.Part(text=prompt)]
            )
            
            # Ajanı çalıştır
            events = runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=user_content
            )
            
            summary = ""
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    summary = event.content.parts[0].text
                    break

            if summary:
                return summary
            
            # if summary:
            #     # Özeti kaydet
            #     await self.save_summary(file_path, summary, content)
            #     self.processed_files.add(file_hash)
            #     return summary
            
        except Exception as e:
            print (f"Özet oluşturulurken hata: {file_path} - {str(e)}")
            return None

    async def process_new_file(self, file_path):
        """Yeni dosyayı işle"""
        try:
            # Dosyanın tamamen yazılmasını bekle
            await asyncio.sleep(1)
            
            if not os.path.exists(file_path):
                return
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return
            
            print(f"\n🔍 Yeni dosya tespit edildi: {file_path}")
            print(f"📊 Dosya boyutu: {file_size} byte")
            print(f"⏳ LLM analiz başlatılıyor...")
            
            # Belgeyi özetle
            summary = await self.summarize_document(file_path)
            
            if summary:
                print(f"\n🤖 LLM ANALİZ SONUCU:")
                print("="*60)
                print(summary)
                print("="*60)
                print(f"✅ Dosya özetlendi!")
            else:
                print(f"❌ LLM analizi başarısız: {file_path}")
                
        except Exception as e:
            print (f"Dosya işlenirken hata: {file_path} - {str(e)}")


class FileWatcher(FileSystemEventHandler):
    def __init__(self, summarizer):
        self.summarizer = summarizer
        super().__init__()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            # Dosyayı kuyruğa ekle (senkron)
            self.summarizer.add_file_to_queue(event.src_path)

async def main():
    """Ana fonksiyon"""
    print("🚀 Proaktif Dosya İzleme ve Özetleme Sistemi Başlatılıyor...")
    print("="*60)
    
    # Özetleyici sistemi oluştur
    summarizer = DocumentSummarizerAgent()
    
    # Event loop'u ayarla
    loop = asyncio.get_running_loop()
    summarizer.set_event_loop(loop)
    
    # Dosya izleyici oluştur
    event_handler = FileWatcher(summarizer)
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    
    # İzlemeyi başlat
    observer.start()
    print(f"👀 Dosya izleme başlatıldı: {WATCH_DIRECTORY}")
    print("📝 .txt dosyaları otomatik olarak özetlenecek...")
    print("🛑 Durdurmak için Ctrl+C'ye basın")
    print("="*60)
    
    try:
        # Dosya işleme task'ını başlat
        file_processor_task = asyncio.create_task(summarizer.process_file_queue())
        
        # Sonsuz döngü - sistem çalışmaya devam etsin
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Sistem durduruldu.")
    finally:
        observer.stop()
        observer.join()
        
        # Task'ı temizle
        if 'file_processor_task' in locals():
            file_processor_task.cancel()
            try:
                await file_processor_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":    
    # Ana sistemi başlat
    asyncio.run(main())