from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools import google_search
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import unicodedata
import os
import logging
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as PlatypusImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from dotenv import load_dotenv

class NoToolNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg.startswith("Warning: there are non-text parts in the response: ['function_call']"):
            return False  # Bu mesajı yoksay
        return True
    
logging.getLogger("google_genai.types").addFilter(NoToolNoiseFilter())

# Set Gemini API Key and Brevo settings
load_dotenv()
# --- Constants ---
APP_NAME = "create_document"
USER_ID = "user1234"
SESSION_ID = "1234"
GEMINI_2_FLASH = "gemini-2.0-flash-lite"

def to_ascii(text):
    """Convert text to ASCII-only characters."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

async def create_image(tool_context: ToolContext):
    """
    Arama sonucuna dayalı olarak Gemini kullanarak bir görsel oluşturur.
    
    Args:
        tool_context: Durum bilgilerini içeren araç bağlamı.
        
    Returns:
        Oluşturulan görselin dosya adı.
    """
    prompt = tool_context.state['summary_search_result']
    #print(f"Özete dayalı görsel oluşturuluyor: {prompt}")
    
    client = genai.Client()
    
    try:
        # Türkçe karakter sorunu engellemek için UTF-8 encoding kullanılıyor
        contents = prompt + " cümlesindeki ana fikri kullanarak bir görsel üret"
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"🤖 [Image Agent] {part.text}")
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                
                # Görseli bir dosyaya kaydet (önce belleğe)
                img_buffer = BytesIO()
                image.save(img_buffer, format="PNG")
                img_data = img_buffer.getvalue()
                
                # Görseli bir artifact olarak kaydet
                filename = "summarized_search_result.png"
                
                await tool_context.save_artifact(
                    filename,
                    types.Part.from_bytes(data=img_data, mime_type="image/png"),
                )
                
                # Ayrıca dosya olarak da kaydet
                with open(filename, "wb") as f:
                    f.write(img_data)
                
                tool_context.state["created_image"] = filename
                return filename
                
    except Exception as e:
        print(f"Görsel oluşturma hatası: {e}")
        import traceback
        print(traceback.format_exc())
        return None

async def create_document(tool_context: ToolContext):
    """
    Görsel ve arama sonuçlarıyla bir PDF belgesi oluşturur.
    Metinler sola hizalı olacak şekilde formatlanır ve Türkçe karakterleri destekler.

    Args:
        tool_context: Durum bilgilerini içeren araç bağlamı.

    Returns:
        Oluşturulan PDF dosyasının adı.
    """
    try:
        # Görseli ve metni al
        image_artifact = await tool_context.load_artifact(tool_context.state['created_image'])
        report_text = tool_context.state["search_result"]

        # Görsel verilerini BytesIO nesnesine yükle
        img_buffer = BytesIO(image_artifact.inline_data.data)
        img = Image.open(img_buffer)

        # PDF belgesi oluştur
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=50,
            rightMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        # --- Türkçe karakter desteği için font çözümü ve metin ön işleme ---
        # Base-14 fontları (Helvetica) kullanılarak Türkçe karakterler Latin-1 karşılıklarına dönüştürülüyor.
        def fix_turkish_chars(text):
            replacements = {
                'ı': 'i', 'İ': 'I', 'ğ': 'g', 'Ğ': 'G',
                'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S',
                'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
            }
            for tr_char, latin_char in replacements.items():
                text = text.replace(tr_char, latin_char)
            return text

        processed_report_text = fix_turkish_chars(report_text)
        processed_topic = fix_turkish_chars(tool_context.state.get("topic", "Rapor"))

        tool_context.state["processed_topic"] = processed_topic

        # --- Sayfa düzeni ve stilleri ---
        styles = getSampleStyleSheet()

        # Metin stili - Sola hizalı (TA_LEFT)
        left_aligned_style = ParagraphStyle(
            name='LeftAlignedStyle',
            parent=styles['Normal'],
            fontName='Helvetica', # Sabit font Helvetica
            alignment=TA_LEFT,
            fontSize=12,
            leading=14,
            firstLineIndent=0, # İlk satır girintisi kaldırıldı
        )

        # Başlık stili
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica', # Sabit font Helvetica
            alignment=TA_LEFT, # Başlık da sola hizalı
            fontSize=16,
            leading=20,
        )

        # --- PDF içeriğini oluştur ---
        flowables = []

        # Başlık ekle
        flowables.append(Paragraph(processed_topic, title_style))
        flowables.append(Spacer(1, 20))

        # Görseli ekle
        width, height = A4
        max_img_width = width - 100
        scale_factor = min(1, max_img_width / img.width)
        img_width = img.width * scale_factor
        img_height = img.height * scale_factor

        img_path = "temp_image.png"
        img.save(img_path)
        flowables.append(PlatypusImage(img_path, width=img_width, height=img_height))
        flowables.append(Spacer(1, 20))

        # Metni paragraf olarak ekle
        paragraphs = processed_report_text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = processed_report_text.split('\n')

        for paragraph in paragraphs:
            if paragraph.strip():
                p = Paragraph(paragraph, left_aligned_style)
                flowables.append(p)
                flowables.append(Spacer(1, 10))

        # PDF'i oluştur
        doc.build(flowables)
        buffer.seek(0)

        # Geçici dosyayı temizle
        if os.path.exists(img_path):
            os.remove(img_path)

        # Artifact olarak kaydet
        filename = "report.pdf"
        await tool_context.save_artifact(
            filename,
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="application/pdf"),
        )

        tool_context.state["created_document"] = filename
        return filename

    except Exception as e:
        import traceback
        print(f"Belge oluşturma hatası: {e}")
        print(traceback.format_exc())
        return None

async def save_document_to_disk(tool_context: ToolContext):
    """
    Oluşturulan PDF belgesini diske kaydeder.
    
    Args:
        tool_context: Durum bilgilerini içeren araç bağlamı.
        
    Returns:
        Kaydedilen dosyanın adı.
    """
    #print("PDF belgesi diske kaydediliyor...")
    
    try:
        # PDF içeriğini al
        filename = tool_context.state["created_document"]
        pdf_artifact = await tool_context.load_artifact(filename)
        
        # PDF'i dosyaya kaydet
        with open(filename, "wb") as f:
            f.write(pdf_artifact.inline_data.data)
        
        return filename
        
    except Exception as e:
        import traceback
        print(f"Belge kaydetme hatası: {e}")
        print(traceback.format_exc())
        return None
    
async def send_mail(tool_context:ToolContext):
    
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    # E-posta gönderim bilgileri
    smtp_server = os.getenv("SMTP_SERVER")  # Brevo SMTP sunucusu
    smtp_port = os.getenv("SMTP_PORT") # Brevo SMTP port (587 TLS, 465 SSL)
    smtp_user = os.getenv("SMTP_USER")  # Brevo'da tanımlı mail adresi
    smtp_password = os.getenv("SMTP_PASSWORD")  # Brevo API
    # Gönderici ve alıcı bilgileri
    from_email = tool_context.state["mail_address"]
    to_email = tool_context.state["mail_address"]  # Göndereceğiniz alıcının e-posta adresi

    # E-posta içeriği
    subject = "Test PDF E-posta"
    body = "Bu bir test e-postasıdır. PDF dosyası ile gönderildi!"

    # E-posta oluşturma
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Eklenecek PDF dosyasının yolu

    pdf_artifact = await tool_context.load_artifact(tool_context.state['created_document'])
    pdf_filename = "report.pdf"

    # PDF dosyasını MIMEBase ile ekleyelim
    part = MIMEBase('application', 'pdf')
    part.set_payload(pdf_artifact.inline_data.data)
    encoders.encode_base64(part)  # Base64 kodlama
    part.add_header('Content-Disposition', f'attachment; filename={pdf_filename}')

    # E-postaya PDF dosyasını ekleyelim
    msg.attach(part)

    # SMTP sunucusuna bağlan ve e-posta gönder
    try:
        # SMTP sunucusuna bağlan
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # TLS şifrelemesi başlat
        server.login(smtp_user, smtp_password)  # Giriş yap

        # E-posta gönder
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)

    except Exception as e:
        print(f"E-posta gönderme sırasında bir hata oluştu: {e}")

    finally:
        # Bağlantıyı kapat
        server.quit()
    return

def before_search_agent_model(callback_context: CallbackContext, llm_request: LlmRequest):
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"🔄 [Callback] Before model call for agent: {agent_name}")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"🔄 [Callback] Inspecting last user message: '{last_user_message}'")

    # Check if the last user message contains "BLOCK"
    if "BLOCK" in last_user_message.upper():
        print("🔄 [Callback] 'BLOCK' keyword found. Skipping LLM call.")
        # Return an LlmResponse to skip the actual LLM call
        callback_context.state["stop_sequential"] = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        print("🔄 [Callback] Proceeding with LLM call.")
        # Return None to allow the (modified) request to go to the LLM
        return None
    
def before_all_agent_model(callback_context: CallbackContext, llm_request: LlmRequest):
    # Eğer BLOCK yakalandıysa veya stop_sequential bayrağı varsa, agent'ı atla
    if callback_context.state["stop_sequential"]:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Bu adım atlandı çünkü akış BLOCK ile durduruldu.")]
            )
        )
    return None

search_agent = LlmAgent(
    name="search",
    model=GEMINI_2_FLASH,
    instruction="""
        Bağlamdaki {topic} hakkında bilgi aramak için google_search aracını kullan.
        Arama sonuçlarını durum değişkenine kaydet.
    """,
    before_model_callback=before_search_agent_model,
    tools=[google_search],
    output_key="search_result"
)

summary_search_agent = LlmAgent(
    name="summary_search",
    model=GEMINI_2_FLASH,
    instruction="""
        {search_result} içindeki arama sonuçlarını tek bir özlü cümleyle özetle.
        Özet, temel bilgileri yakalamalıdır.
        Özeti durum değişkenine kaydet.
    """,
    before_model_callback=before_all_agent_model,
    output_key="summary_search_result"
)

create_image_agent = LlmAgent(
    name="create_image",
    model=GEMINI_2_FLASH,
    instruction="""
        {summary_search_result} içindeki özeti kullanarak bir görsel oluştur.
        Bu özet görselle görsel olarak temsil edilmelidir.
        Görseli oluşturmak için create_image aracını kullan.
        Kullanıcıdan herhangi bir onay ya da ek bir bilgi bekleme!
    """,
    before_model_callback=before_all_agent_model,
    tools=[create_image]
)

create_document_agent = LlmAgent(
    name="create_document",
    model=GEMINI_2_FLASH,
    instruction="""
        Şunları kullanarak bir PDF belgesi oluştur:
        1. {created_image} içindeki görsel
        2. {search_result} içindeki arama sonuçları
        
        PDF'i oluşturmak için create_document aracını kullan.
        PDF'i sadece artifact olarak kaydet, disk kaydetme işlemi ayrı bir agent tarafından yapılacak.
        Kullanıcıdan herhangi bir onay bekleme.
    """,
    before_model_callback=before_all_agent_model,
    tools=[create_document]
)

save_document_to_disk_agent = LlmAgent(
    name="save_document_to_disk",
    model=GEMINI_2_FLASH,
    instruction="""
        {created_document} içindeki PDF belgesini diske kaydet.
        
        PDF'i diske kaydetmek için save_document_to_disk aracını kullan.
        Kullanıcıdan herhangi bir onay bekleme.
    """,
    before_model_callback=before_all_agent_model,
    tools=[save_document_to_disk]
)

send_mail_agent = LlmAgent(
    name="send_mail",
    model=GEMINI_2_FLASH,
    instruction="""
        {created_document} içindeki pdf belgesini mail olarak gönder.

        Mail gönderebilmek için send_mail aracını kullan. Mail göndermek için gerekli bütün bilgiler bu aracın içinde olacak.
        
        Kullanıcıdan herhangi bir onay ya da ek bir bilgi bekleme!
    """,
    before_model_callback=before_all_agent_model,
    tools=[send_mail],
)

parallel_agents = ParallelAgent(
    name="parallel_tasks",
    sub_agents=[save_document_to_disk_agent, send_mail_agent]
)

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

#from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters
from utils.custom_adk_patches import CustomMCPToolset as MCPToolset

async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server."""
    tools = MCPToolset (
        # Use StdioServerParameters for local process communication
        connection_params=StdioServerParameters(
            command='python', # Command to run the server
            args=[
                "C:\Zafer\Kod\Agents\MCPServer.py"],
        )
    )
    # print(f"Fetched {len(tools.get_tools().)} tools from MCP server.")
    # print([tool.name for tool in tools])
    mcp_client_agent = LlmAgent(
        model=GEMINI_2_FLASH, # Adjust model name if needed based on availability
        name='mcp_client',
            instruction = """
                Use only and only the {processed_topic} variable.
                Do NOT USE, SEE, or INTERPRET any other state variable, search result, summary, image, or document.
                WITHOUT MAKING ANY CHANGES WHATSOEVER to the words or their order in {processed_topic}, write them exactly as they are to the file C:\\Zafer\\Kod\\Agents\\Temp\\List1.txt using the append_to_file tool.
                Before writing, you MUST call the get_current_datetime tool and use its result for the date and time. 
                When calling get_current_datetime, pass any dummy string as its parameter (for example: "now" or "dummy").
                Always prepend the returned date and time in the format [YYYY.MM.DD HH:MM] before the text. For example: [2025.05.22 18:02] {processed_topic}
                Do not perform any other operation, do not look at or process any other information.
                IMPORTANT: Never generate the date and time yourself. Always use the value returned by the get_current_datetime tool.
                """,
        before_model_callback=before_all_agent_model,
        tools=[tools],
    )
    return mcp_client_agent, tools

async def call_agent(query):
    """
    Aracıyı bir sorguyla çağır.
    
    Args:
        query: Kullanıcının arama yapacağı konu/sorgu.
    """
    
    mcp_client_agent, tools = await get_agent_async()

    main_agent = SequentialAgent(
        name="document_creator",
        sub_agents=[search_agent, summary_search_agent, create_image_agent, create_document_agent, parallel_agents, mcp_client_agent]
    )

    runner = Runner(
        agent=main_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    initial_state = {
        "mail_address": os.getenv("MAIL_ADDRESS"),
        "topic": query,
        "processed_topic":"",
        "search_result": "",
        "summary_search_result": "",
        "created_image": "",
        "created_document": "",
        "stop_sequential":False,
    }
    
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )

    # Kullanıcı içeriğini oluştur
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    # Aracıyı çalıştır
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    
    # Olayları işle
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            response = event.content.parts[0].text
            print(f"🤖 [Agent] {event.author}: {response}")

    await tools.close()

if __name__ == "__main__":
    import asyncio
    #asyncio.run(call_agent("Yapay zekanin gelecekte getirecegi olasi tehditler"))
    asyncio.run(call_agent("Kuresel iklimdeki degisikliklerin gelecekte olusturacagi tehditler"))