#!/usr/bin/env python3
# ==============================================================================
#                  Simple MCP Server - Demo Implementation
# ==============================================================================

"""A simple MCP server with practical utility tools.

This server demonstrates:
- Text manipulation tools
- Math operations
- Data formatting utilities
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from openmcp import MCPServer, tool

# Suppress logs for clean output
for logger_name in ("mcp", "httpx", "uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

server = MCPServer("simple-utility-server")


with server.binding():

    @tool(description="Reverse a string")
    def reverse_text(text: str) -> str:
        """Takes a string and returns it reversed."""
        return text[::-1]

    @tool(description="Count words in a text")
    def count_words(text: str) -> dict[str, int]:
        """Analyzes text and returns word count statistics."""
        words = text.split()
        return {
            "total_words": len(words),
            "total_characters": len(text),
            "total_characters_no_spaces": len(text.replace(" ", "")),
        }

    @tool(description="Calculate factorial of a number")
    def factorial(n: int) -> int:
        """Calculates the factorial of a non-negative integer."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @tool(description="Get current timestamp information")
    async def get_timestamp() -> dict[str, str | int]:
        """Returns current timestamp in multiple formats."""
        now = datetime.now()
        return {
            "iso_format": now.isoformat(),
            "unix_timestamp": int(now.timestamp()),
            "human_readable": now.strftime("%Y-%m-%d %H:%M:%S"),
            "year": now.year,
            "month": now.month,
            "day": now.day,
        }

    @tool(description="Convert temperature between Celsius and Fahrenheit")
    def convert_temperature(value: float, from_unit: str) -> dict[str, float]:
        """Converts temperature between C and F.

        Args:
            value: The temperature value
            from_unit: Either 'C' or 'F'
        """
        from_unit = from_unit.upper()
        if from_unit == "C":
            fahrenheit = (value * 9 / 5) + 32
            return {"celsius": value, "fahrenheit": round(fahrenheit, 2)}
        elif from_unit == "F":
            celsius = (value - 32) * 5 / 9
            return {"celsius": round(celsius, 2), "fahrenheit": value}
        else:
            raise ValueError("from_unit must be 'C' or 'F'")

    @tool(description="Create a simple list from comma-separated values")
    def parse_csv_line(line: str) -> list[str]:
        """Splits a CSV line into a list of values."""
        return [item.strip() for item in line.split(",")]


async def main() -> None:
    """Start the MCP server on streamable-http transport."""
    print("🚀 Starting Simple Utility Server...")
    print("📡 Transport: streamable-http")
    print("🔧 Available tools: reverse_text, count_words, factorial, get_timestamp, convert_temperature, parse_csv_line")
    print()
    await server.serve(transport="streamable-http", verbose=False, log_level="critical")


if __name__ == "__main__":
    asyncio.run(main())
