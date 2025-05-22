
def lengthOfLongestSubstring(s):
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    max_length = 0
    substring = ""
    start_idx = 0
    while start_idx < len(s):
        string = s[start_idx:]
        for i, x in enumerate(string):
            substring += x
            if len(substring) == len(set((list(substring)))):

                if len(set((list(substring)))) > max_length:

                    max_length = len(substring)

        start_idx += 1
        substring = ""


    return max_length
import pytest

def test_empty_string():
    """Test with an empty string."""
    assert lengthOfLongestSubstring("") == 0

def test_single_character_string():
    """Test with a single character string."""
    assert lengthOfLongestSubstring("a") == 1

def test_string_with_unique_characters():
    """Test with a string containing all unique characters."""
    assert lengthOfLongestSubstring("abcdefg") == 7

def test_string_with_repeating_characters():
    """Test with a string containing repeating characters."""
    assert lengthOfLongestSubstring("abcabcbb") == 3

def test_string_with_all_same_characters():
    """Test with a string containing all same characters."""
    assert lengthOfLongestSubstring("bbbbbb") == 1

def test_string_with_leading_and_trailing_spaces():
    """Test with a string containing leading and trailing spaces."""
    assert lengthOfLongestSubstring("  abc  ") == 3

def test_string_with_internal_spaces():
    """Test with a string containing internal spaces."""
    assert lengthOfLongestSubstring("ab  cd") == 4

def test_string_with_mixed_case_characters():
    """Test with a string containing mixed case characters."""
    assert lengthOfLongestSubstring("aAbBcC") == 6

def test_string_with_special_characters():
    """Test with a string containing special characters."""
    assert lengthOfLongestSubstring("abc!@#") == 6

def test_string_with_unicode_characters():
    """Test with a string containing Unicode characters."""
    assert lengthOfLongestSubstring("你好世界") == 4

def test_long_string():
    """Test with a very long string."""
    long_string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert lengthOfLongestSubstring(long_string) == 62

def test_string_with_repeated_substring():
    """Test with a string that has repeated substrings."""
    assert lengthOfLongestSubstring("tmmzuxt") == 5

def test_non_string_input():
    """Test with a non-string input to raise TypeError."""
    with pytest.raises(TypeError):
        lengthOfLongestSubstring(123)

def test_string_with_numbers():
    """Test with a string including numbers."""
    assert lengthOfLongestSubstring("1234567") == 7
