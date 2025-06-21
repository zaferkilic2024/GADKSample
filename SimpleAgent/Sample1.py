from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, ToolContext
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

APP_NAME = "search_agent"
USER_ID = "user1234"
SESSION_ID = "1234"
GEMINI_2_FLASH = "gemini-2.0-flash-lite"

def before_main_agent(callback_context: CallbackContext, llm_request: LlmRequest):
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text

    if (last_user_message and "BLOCK" in last_user_message.upper()) or callback_context.state["stop_sequential"] == True:
        callback_context.state["stop_sequential"] = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    return None


def text_statistics(text: str) -> dict:
    import re
    paragraphs = [p for p in text.split('\n') if p.strip()]
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    words = re.findall(r'\b\w+\b', text)
    # Sadece satır sonlarını hariç tut, diğer tüm karakterleri say
    characters = [c for c in text if c not in ('\n', '\r')]
    return {
        "character_count": len(characters),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs)
    }

show_statistic = LlmAgent(
    name="show_statistic",
    model=GEMINI_2_FLASH,
    instruction=
        """
        {search_result1} metnini analiz et ve her bir istatistiği doğru bir şekilde hesapla. Bu hesap için text_statistics aracını kullan. 
        Türkçe sonuç üret. Son paragrafı analiz sonuçlarından hemen önce göster.
        📊 {topic} arama sonucu istatistikleri
        ================================
        📝 Karakter sayısı: [text_statistics aracından dönen character_count değeri]
        🔤 Kelime sayısı: [text_statistics aracından dönen word_count değeri]
        📄 Cümle sayısı: [text_statistics aracından dönen sentence_count değeri]
        📋 Paragraf sayısı: [text_statistics aracından dönen paragraph_count değeri]
        """,
    tools=[text_statistics],
    before_model_callback=before_main_agent,
    output_key="statistic"  
)

search_agent1 = LlmAgent(
    name="search_agent1",
    model=GEMINI_2_FLASH,
    instruction=
        """
        {topic} içindeki konu hakkında google_search aracını kullanarak arama yap.
        """,
    tools=[google_search],
    before_model_callback=before_main_agent,
    output_key="search_result1"
)

search_agent2 = LlmAgent(
    name="search_agent2",
    model=GEMINI_2_FLASH,
    instruction=
        """
        {topic} içindeki konu hakkında google_search aracını kullanarak arama yap.
        """,
    tools=[google_search],
    before_model_callback=before_main_agent,
    output_key="search_result2"
)

parallel_agents= ParallelAgent(
    name="parallel_tasks",
    sub_agents=[search_agent1, search_agent2]
)

main_agent = SequentialAgent(
    name="main_agent",
    sub_agents=[parallel_agents, show_statistic]
)

async def call_agent(query):
    """
    Aracıyı bir sorguyla çağır.
    
    Args:
        query: Kullanıcının arama yapacağı konu/sorgu.
    """
    session_service = InMemorySessionService()

    initial_state = {
        "topic": query,
        "stop_sequential": False,
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
    asyncio.run(call_agent("Kuresel BLOCK iklimdeki degisikliklerin gelecekte olusturacagi tehditler"))