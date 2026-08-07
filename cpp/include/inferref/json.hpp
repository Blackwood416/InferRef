// Minimal dependency-free JSON parser for the native example engines.
//
// The SYCL engine reads a testcase manifest to select its executable contract.
// Earlier revisions located the contract with raw string scanning, which is
// brittle against strings that merely mention "contracts".  This header
// provides a small recursive-descent parser with proper string escapes,
// duplicate-key-last-wins semantics, and position-annotated errors, without
// pulling in a third-party dependency.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferref
{
namespace json
{

struct Value;

using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Type
{
    kNull,
    kBool,
    kNumber,
    kString,
    kArray,
    kObject,
};

struct Value
{
    Type type = Type::kNull;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    Array array;
    Object object;

    bool IsNull() const { return type == Type::kNull; }
    bool IsBool() const { return type == Type::kBool; }
    bool IsNumber() const { return type == Type::kNumber; }
    bool IsString() const { return type == Type::kString; }
    bool IsArray() const { return type == Type::kArray; }
    bool IsObject() const { return type == Type::kObject; }

    bool Has(const std::string &key) const { return object.count(key) != 0; }
    const Value &At(const std::string &key) const { return object.at(key); }
};

class ParseError : public std::runtime_error
{
public:
    explicit ParseError(const std::string &message) : std::runtime_error(message) {}
};

// Parse one JSON document.  Throws json::ParseError on malformed input.
Value Parse(const std::string &text);

// Reads a JSON string array, returning its elements in order.  Throws
// json::ParseError with `context` in the message when the value is not an
// array of strings.
std::vector<std::string> StringArray(const Value &value, const std::string &context);

// Locates the testcase executable-contract list: the top-level "contracts"
// field first, then "requirements.contracts".  Returns an empty vector when
// neither exists.
std::vector<std::string> DeclaredContracts(const Value &root);

namespace detail
{

class Parser
{
public:
    explicit Parser(const std::string &text) : text_(text), position_(0) {}

    Value ParseDocument()
    {
        SkipWhitespace();
        Value value = ParseValue();
        SkipWhitespace();
        if (position_ != text_.size())
            Fail("trailing content after JSON document");
        return value;
    }

private:
    const std::string &text_;
    std::size_t position_;

    [[noreturn]] void Fail(const std::string &message) const
    {
        throw ParseError(message + " at offset " + std::to_string(position_));
    }

    char Peek() const { return position_ < text_.size() ? text_[position_] : '\0'; }

    void SkipWhitespace()
    {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])))
            ++position_;
    }

    void Expect(char expected)
    {
        if (Peek() != expected)
            Fail(std::string("expected '") + expected + "'");
        ++position_;
    }

    bool ConsumeLiteral(const char *literal)
    {
        const std::size_t length = std::string(literal).size();
        if (text_.compare(position_, length, literal) != 0)
            return false;
        position_ += length;
        return true;
    }

    Value ParseValue()
    {
        SkipWhitespace();
        if (position_ >= text_.size())
            Fail("unexpected end of input");
        const char current = text_[position_];
        if (current == '{')
            return ParseObject();
        if (current == '[')
            return ParseArray();
        if (current == '"')
            return ParseString();
        if (current == 't' && ConsumeLiteral("true"))
            return MakeBool(true);
        if (current == 'f' && ConsumeLiteral("false"))
            return MakeBool(false);
        if (current == 'n' && ConsumeLiteral("null"))
            return Value();
        if (current == '-' || std::isdigit(static_cast<unsigned char>(current)))
            return ParseNumber();
        Fail("unexpected character");
    }

    static Value MakeBool(bool value)
    {
        Value result;
        result.type = Type::kBool;
        result.boolean = value;
        return result;
    }

    Value ParseObject()
    {
        Expect('{');
        Value result;
        result.type = Type::kObject;
        SkipWhitespace();
        if (Peek() == '}')
        {
            ++position_;
            return result;
        }
        while (true)
        {
            SkipWhitespace();
            if (Peek() != '"')
                Fail("expected object key string");
            const std::string key = ParseString().string;
            SkipWhitespace();
            Expect(':');
            const Value value = ParseValue();
            // Duplicate keys: last one wins, matching Python's json.loads.
            result.object[key] = std::move(value);
            SkipWhitespace();
            if (Peek() == ',')
            {
                ++position_;
                continue;
            }
            if (Peek() == '}')
            {
                ++position_;
                return result;
            }
            Fail("expected ',' or '}' in object");
        }
    }

    Value ParseArray()
    {
        Expect('[');
        Value result;
        result.type = Type::kArray;
        SkipWhitespace();
        if (Peek() == ']')
        {
            ++position_;
            return result;
        }
        while (true)
        {
            result.array.push_back(ParseValue());
            SkipWhitespace();
            if (Peek() == ',')
            {
                ++position_;
                continue;
            }
            if (Peek() == ']')
            {
                ++position_;
                return result;
            }
            Fail("expected ',' or ']' in array");
        }
    }

    Value ParseString()
    {
        Expect('"');
        Value result;
        result.type = Type::kString;
        std::string &out = result.string;
        while (true)
        {
            if (position_ >= text_.size())
                Fail("unterminated string");
            const char current = text_[position_++];
            if (current == '"')
                return result;
            if (current != '\\')
            {
                if (static_cast<unsigned char>(current) < 0x20)
                    Fail("unescaped control character in string");
                out.push_back(current);
                continue;
            }
            if (position_ >= text_.size())
                Fail("unterminated escape sequence");
            const char escape = text_[position_++];
            switch (escape)
            {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': AppendUnicodeEscape(out); break;
            default: Fail("invalid escape sequence");
            }
        }
    }

    void AppendUnicodeEscape(std::string &out)
    {
        const unsigned int first = ReadHexQuad();
        if (first >= 0xD800 && first <= 0xDBFF)
        {
            // High surrogate: expect a following low surrogate.
            if (position_ + 1 >= text_.size() || text_[position_] != '\\' ||
                text_[position_ + 1] != 'u')
                Fail("unpaired high surrogate");
            position_ += 2;
            const unsigned int low = ReadHexQuad();
            if (low < 0xDC00 || low > 0xDFFF)
                Fail("invalid low surrogate");
            AppendUtf8(out, 0x10000 + ((first - 0xD800) << 10) + (low - 0xDC00));
            return;
        }
        if (first >= 0xDC00 && first <= 0xDFFF)
            Fail("unpaired low surrogate");
        AppendUtf8(out, first);
    }

    unsigned int ReadHexQuad()
    {
        if (position_ + 4 > text_.size())
            Fail("truncated \\u escape");
        unsigned int value = 0;
        for (std::size_t i = 0; i < 4; ++i)
        {
            const char current = text_[position_++];
            const int digit = HexValue(current);
            if (digit < 0)
                Fail("invalid \\u escape");
            value = (value << 4) | static_cast<unsigned int>(digit);
        }
        return value;
    }

    static int HexValue(char current)
    {
        if (current >= '0' && current <= '9')
            return current - '0';
        if (current >= 'a' && current <= 'f')
            return current - 'a' + 10;
        if (current >= 'A' && current <= 'F')
            return current - 'A' + 10;
        return -1;
    }

    static void AppendUtf8(std::string &out, unsigned int codepoint)
    {
        if (codepoint <= 0x7F)
        {
            out.push_back(static_cast<char>(codepoint));
        }
        else if (codepoint <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        else if (codepoint <= 0xFFFF)
        {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    Value ParseNumber()
    {
        const std::size_t start = position_;
        if (Peek() == '-')
            ++position_;
        if (position_ >= text_.size())
            Fail("invalid number");
        if (Peek() == '0')
        {
            ++position_;
        }
        else if (std::isdigit(static_cast<unsigned char>(Peek())))
        {
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])))
                ++position_;
        }
        else
        {
            Fail("invalid number");
        }
        if (Peek() == '.')
        {
            ++position_;
            if (position_ >= text_.size() ||
                !std::isdigit(static_cast<unsigned char>(text_[position_])))
                Fail("invalid fractional number");
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])))
                ++position_;
        }
        if (Peek() == 'e' || Peek() == 'E')
        {
            ++position_;
            if (Peek() == '+' || Peek() == '-')
                ++position_;
            if (position_ >= text_.size() ||
                !std::isdigit(static_cast<unsigned char>(text_[position_])))
                Fail("invalid exponent");
            while (position_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[position_])))
                ++position_;
        }
        Value result;
        result.type = Type::kNumber;
        result.number = std::strtod(text_.substr(start, position_ - start).c_str(), nullptr);
        return result;
    }
};

} // namespace detail

inline Value Parse(const std::string &text)
{
    detail::Parser parser(text);
    return parser.ParseDocument();
}

inline std::vector<std::string> StringArray(const Value &value, const std::string &context)
{
    std::vector<std::string> result;
    if (!value.IsArray())
        throw ParseError(context + " must be a JSON string array");
    result.reserve(value.array.size());
    for (const Value &item : value.array)
    {
        if (!item.IsString())
            throw ParseError(context + " must contain only strings");
        result.push_back(item.string);
    }
    return result;
}

inline std::vector<std::string> DeclaredContracts(const Value &root)
{
    if (root.IsObject() && root.Has("contracts"))
        return StringArray(root.At("contracts"), "contracts");
    if (root.IsObject() && root.Has("requirements") &&
        root.At("requirements").IsObject() &&
        root.At("requirements").Has("contracts"))
        return StringArray(root.At("requirements").At("contracts"),
                           "requirements.contracts");
    return {};
}

} // namespace json
} // namespace inferref
