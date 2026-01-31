"""Quick test of the RAG system."""
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from config import get_config
from src.ingestion import TavilySearchClient
from src.utils import CostTracker

print("\n" + "="*60)
print("TESTING AGENTIC RAG SYSTEM")
print("="*60)

try:
    # Initialize
    print("\n1. Initializing...")
    config = get_config()
    cost_tracker = CostTracker()
    
    openai_client = OpenAI(api_key=config.api.openai_api_key)
    tavily_client = TavilySearchClient(
        api_key=config.api.tavily_api_key,
        cost_tracker=cost_tracker
    )
    print("   ✓ Clients initialized")
    
    # Test search
    print("\n2. Testing web search...")
    test_query = "Python programming basics"
    results = tavily_client.search(test_query, max_results=3)
    print(f"   ✓ Found {len(results)} sources")
    
    for i, result in enumerate(results[:2], 1):
        print(f"   - {result.title[:50]}... ({result.source_type})")
    
    # Test cost tracking
    print("\n3. Checking cost tracking...")
    costs = cost_tracker.get_session_costs()
    print(f"   ✓ Session cost: ${costs['total']:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nSystem is working! Ready for full test with: python main.py")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
