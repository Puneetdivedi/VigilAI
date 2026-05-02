"""
Unit tests for LangGraph agents.
"""
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.graph import build_graph

def test_graph_structure():
    """Test that the graph builds without error."""
    graph = build_graph()
    assert graph is not None
    # Check nodes
    node_names = [n for n in graph.nodes]
    assert "anomaly_detector" in node_names
    assert "rag_retriever" in node_names
    assert "report_writer" in node_names
