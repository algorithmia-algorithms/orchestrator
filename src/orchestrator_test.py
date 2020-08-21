from . import orchestrator

def test_orchestrator():
    assert orchestrator.apply("Jane") == "hello Jane"
