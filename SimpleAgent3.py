from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

import logging
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

def before_main_agent(callback_context: CallbackContext, llm_request: LlmRequest):
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text

    if "BLOCK" in last_user_message.upper():
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    return None

show_statistic = LlmAgent(
    name="show_statistic",
    model=GEMINI_2_FLASH,
    instruction=
        """
        Arama sonucunu şu formatta göster:
        📊 {topic} arama sonucu istatistikleri
        ================================
        📝 Kelime sayısı: [metin içindeki toplam kelime sayısı]
        🔤 Harf sayısı: [metin içindeki toplam harf sayısı (boşluklar hariç)]
        📄 Karakter sayısı: [metin içindeki toplam karakter sayısı (boşluklar dahil)]
        📋 Paragraf sayısı: [metin içindeki paragraf sayısı]
    
        Metni analiz et ve her bir istatistiği doğru bir şekilde hesapla.
        """,
    output_key="statistic"
)

search_agent = LlmAgent(
    name="search_agent",
    model=GEMINI_2_FLASH,
    instruction=
        """
        {topic} içindeki konu hakkında google_search aracını kullanarak arama yap.
        """,
    tools=[google_search],
    before_model_callback=before_main_agent,
    output_key="search_result"
)

main_agent = SequentialAgent(
    name="main_agent",
    sub_agents=[search_agent, show_statistic]
)

session_service = InMemorySessionService()

async def call_agent(query):
    """
    Aracıyı bir sorguyla çağır.
    
    Args:
        query: Kullanıcının arama yapacağı konu/sorgu.
    """
    initial_state = {
        "topic": query,
    }
    
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )

    runner = Runner(
        agent=main_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Kullanıcı içeriğini oluştur
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    # Aracıyı çalıştır
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    
    # Olayları işle
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            response = event.content.parts[0].text
            print(f"🤖 [Agent] {event.author}:\n{response}")

if __name__ == "__main__":
    import asyncio
    #asyncio.run(call_agent("Yapay zekanin gelecekte getirecegi olasi tehditler"))
    asyncio.run(call_agent("Kuresel iklimdeki degisikliklerin gelecekte olusturacagi tehditler"))