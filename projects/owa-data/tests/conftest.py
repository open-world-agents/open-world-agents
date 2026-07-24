#!/usr/bin/env python3
"""Pytest configuration and fixtures."""

import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to print recorded properties after each test."""
    outcome = yield
    report = outcome.get_result()

    # Only print for passed tests in the call phase (not setup/teardown)
    if call.when == "call" and hasattr(report, "user_properties") and report.user_properties:
        print(f"\n📊 Properties for {item.name}:")
        for key, value in report.user_properties:
            print(f"  {key}: {value}")
