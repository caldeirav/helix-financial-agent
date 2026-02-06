"""
Entry point for Streamlit UI. Run from project root:
  uv run streamlit run app.py
  # or: streamlit run app.py
"""
from helix_financial_agent.app.streamlit_app import main

if __name__ == "__main__":
    main()
