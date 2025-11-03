"""
Tests for the EasyDict class.
"""

from pydantic import BaseModel

from owa.core.utils.easydict import EasyDict


class TestEasyDictBasics:
    """Test basic EasyDict functionality."""

    def test_empty_initialization(self):
        """Test creating an empty EasyDict."""
        d = EasyDict()
        assert len(d) == 0

    def test_initialization_with_dict(self):
        """Test creating EasyDict from a dictionary."""
        d = EasyDict({"key": "value", "number": 42})
        assert d["key"] == "value"
        assert d["number"] == 42
        assert len(d) == 2

    def test_initialization_with_kwargs(self):
        """Test creating EasyDict with keyword arguments."""
        d = EasyDict(key="value", number=42)
        assert d["key"] == "value"
        assert d["number"] == 42

    def test_initialization_with_both(self):
        """Test creating EasyDict with both dict and kwargs."""
        d = EasyDict({"key1": "value1"}, key2="value2")
        assert d["key1"] == "value1"
        assert d["key2"] == "value2"


class TestEasyDictAttributeAccess:
    """Test attribute-style access functionality."""

    def test_attribute_access(self):
        """Test accessing values as attributes."""
        d = EasyDict({"name": "test", "value": 123})
        assert d.name == "test"
        assert d.value == 123

    def test_attribute_assignment(self):
        """Test setting values via attributes."""
        d = EasyDict()
        d.name = "test"
        d.value = 123
        assert d["name"] == "test"
        assert d["value"] == 123

    def test_dict_assignment(self):
        """Test setting values via dictionary syntax."""
        d = EasyDict()
        d["name"] = "test"
        d["value"] = 123
        assert d.name == "test"
        assert d.value == 123


class TestEasyDictNested:
    """Test nested dictionary functionality."""

    def test_nested_dict_access(self):
        """Test accessing nested dictionaries as attributes."""
        d = EasyDict({"database": {"host": "localhost", "port": 5432}})
        assert d.database.host == "localhost"
        assert d.database.port == 5432
        assert isinstance(d.database, EasyDict)

    def test_nested_dict_assignment(self):
        """Test assigning nested dictionaries."""
        d = EasyDict()
        d.database = {"host": "localhost", "port": 5432}
        assert d.database.host == "localhost"
        assert d.database.port == 5432
        assert isinstance(d.database, EasyDict)

    def test_deeply_nested_access(self):
        """Test deeply nested dictionary access."""
        d = EasyDict({"level1": {"level2": {"level3": {"value": "deep"}}}})
        assert d.level1.level2.level3.value == "deep"


class TestEasyDictLists:
    """Test list and tuple handling."""

    def test_list_with_dicts(self):
        """Test lists containing dictionaries."""
        d = EasyDict({"servers": [{"name": "web1", "ip": "192.168.1.1"}, {"name": "web2", "ip": "192.168.1.2"}]})
        assert d.servers[0].name == "web1"
        assert d.servers[0].ip == "192.168.1.1"
        assert d.servers[1].name == "web2"
        assert isinstance(d.servers[0], EasyDict)

    def test_tuple_with_dicts(self):
        """Test tuples containing dictionaries."""
        d = EasyDict()
        d.tuple_item = ({"key": "value1"}, {"key": "value2"})
        assert isinstance(d.tuple_item, tuple)
        assert d.tuple_item[0].key == "value1"
        assert d.tuple_item[1].key == "value2"
        assert isinstance(d.tuple_item[0], EasyDict)

    def test_list_with_primitives(self):
        """Test lists containing primitive values."""
        d = EasyDict({"numbers": [1, 2, 3, 4, 5]})
        assert d.numbers == [1, 2, 3, 4, 5]
        assert isinstance(d.numbers, list)


class TestEasyDictMethods:
    """Test dictionary methods."""

    def test_update_method(self):
        """Test the update method."""
        d = EasyDict({"key1": "value1"})
        d.update({"key2": "value2"})
        assert d.key1 == "value1"
        assert d.key2 == "value2"

    def test_update_with_kwargs(self):
        """Test update with keyword arguments."""
        d = EasyDict({"key1": "value1"})
        d.update(key2="value2", key3="value3")
        assert d.key1 == "value1"
        assert d.key2 == "value2"
        assert d.key3 == "value3"

    def test_pop_method(self):
        """Test the pop method."""
        d = EasyDict({"key1": "value1", "key2": "value2"})
        value = d.pop("key1")
        assert value == "value1"
        assert "key1" not in d
        assert not hasattr(d, "key1")
        assert d.key2 == "value2"

    def test_pop_with_default(self):
        """Test pop with default value."""
        d = EasyDict({"key1": "value1"})
        value = d.pop("nonexistent", "default")
        assert value == "default"
        assert len(d) == 1


class TestEasyDictPydanticIntegration:
    """Test Pydantic integration."""

    def test_pydantic_model_with_easydict(self):
        """Test using EasyDict in a Pydantic model."""

        class Config(BaseModel):
            settings: EasyDict

        config_data = {"settings": {"host": "localhost", "port": 8080}}
        config = Config(**config_data)

        assert isinstance(config.settings, EasyDict)
        assert config.settings.host == "localhost"
        assert config.settings.port == 8080

    def test_pydantic_validation(self):
        """Test that Pydantic validation works with EasyDict."""

        class AppConfig(BaseModel):
            database: EasyDict
            cache: EasyDict

        config = AppConfig(database={"host": "localhost", "port": 5432}, cache={"enabled": True, "ttl": 3600})

        assert config.database.host == "localhost"
        assert config.cache.enabled is True


class TestEasyDictEdgeCases:
    """Test edge cases and special scenarios."""

    def test_overwrite_existing_key(self):
        """Test overwriting an existing key."""
        d = EasyDict({"key": "old_value"})
        d.key = "new_value"
        assert d.key == "new_value"
        assert d["key"] == "new_value"

    def test_mixed_access_patterns(self):
        """Test mixing attribute and dictionary access."""
        d = EasyDict()
        d["attr_key"] = "value1"
        d.dict_key = "value2"
        assert d.attr_key == "value1"
        assert d["dict_key"] == "value2"

    def test_none_values(self):
        """Test handling None values."""
        d = EasyDict({"key": None})
        assert d.key is None
        assert d["key"] is None

    def test_boolean_values(self):
        """Test handling boolean values."""
        d = EasyDict({"enabled": True, "disabled": False})
        assert d.enabled is True
        assert d.disabled is False

    def test_numeric_values(self):
        """Test handling various numeric types."""
        d = EasyDict({"integer": 42, "float": 3.14, "negative": -10})
        assert d.integer == 42
        assert d.float == 3.14
        assert d.negative == -10

    def test_empty_nested_dict(self):
        """Test empty nested dictionaries."""
        d = EasyDict({"empty": {}})
        assert isinstance(d.empty, EasyDict)
        assert len(d.empty) == 0

    def test_empty_list(self):
        """Test empty lists."""
        d = EasyDict({"list_item": []})
        assert d.list_item == []
        assert isinstance(d.list_item, list)
