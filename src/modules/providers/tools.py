from datetime import datetime


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone."""
    # Mock implementation - simplified version
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(operation: str, a: float, b: float) -> str:
    """Perform basic arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"
    
    result = operations[operation](a, b)
    return str(result)