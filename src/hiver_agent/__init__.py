"""AI customer-support agent for a single Twitter brand handle.

Submodules are imported by path rather than re-exported here, so that
`import hiver_agent` stays cheap -- eagerly re-exporting RetrievalIndex would
drag in torch and sentence-transformers on every import, including in scripts
that never touch the index.

    from hiver_agent.classifier import classify
    from hiver_agent.retrieval import RetrievalIndex
"""

__version__ = "0.1.0"