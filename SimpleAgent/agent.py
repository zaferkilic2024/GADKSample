from google.adk.agents import Agent, LlmAgent
from google.adk.runners import Runner
from google.adk.tools import google_search, agent_tool
from google.adk.sessions import InMemorySessionService
from google.genai import types

import yfinance as yf
import asyncio

from dotenv import load_dotenv
load_dotenv()

def get_stock_price(symbol: str):
    """
    Retrieves the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period="1d")
        if not historical_data.empty:
            current_price = historical_data['Close'].iloc[-1]
            return current_price
        else:
            return None
    except Exception as e:
        print(f"Error retrieving stock price for {symbol}: {e}")
        return None
    
search_agent = Agent (
    model='gemini-2.0-flash',
    name='search_agent',
    instruction="Bir şirket adı verildiğinde, resmi hisse senedi sembolünü google_search kullanarak bul. Yalnızca hisse senedi sembolüyle yanıt ver.",
    tools=[google_search],
    output_key="ticker_symbol"
)

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='stock_agent',
    instruction=
        """
        Sen bir hisse senedi fiyatlarını getiren bir agentsin.
        Eğer bir hisse senedi sembolü (ticker) verilmişse, mevcut fiyatı get_stock_price aracını kullanarak getir.
        Eğer sadece bir şirket adı verilmişse, önce search_agent aracını kullanarak Google'da arama yap ve doğru hisse senedi sembolünü bul. Ardından hisse fiyatını getir.
        Eğer verilen sembol geçersizse ya da veri alınamıyorsa, kullanıcıya hisse fiyatının bulunamadığını bildir.
        """,
    tools=[get_stock_price, agent_tool.AgentTool(agent=search_agent)],
)