"""
CUDA Source-Level Frontend

This module provides source-to-source translation from CUDA to RIPPLE.
It parses CUDA source code, builds an AST, transforms it to the AIR,
and generates RIPPLE C code.

Architecture:
    CUDA Source -> Lexer -> Parser -> CUDA AST -> AIR Builder -> AIR
                                                                  |
                                                                  v
                                            RIPPLE C Code <- RIPPLE Generator
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Iterator
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.semantic_model import (
    AIRNode, AIRFunction, AIRVariable, AIRType, AIRExpression,
    AIRThreadIndex, AIRLoop, AIRConditional, AIRMemoryOp,
    AIRTranslationUnit, AIRSynchronization, AIRShuffleOp, AIRReductionOp,
    CUDABuiltinAccess, CUDADim3, CUDAMemorySpace, CUDASharedMemory,
    CUDAKernelLaunch, RIPPLEBlockShape, RIPPLEProcessingElement,
    TranslationContext, HexagonConfig
)
from core.translation_rules import TranslationRuleEngine, infer_block_shape


# =============================================================================
# Token Types
# =============================================================================

class TokenType(Enum):
    # Literals
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    CHAR = auto()
    
    # CUDA Keywords
    GLOBAL = auto()       # __global__
    DEVICE = auto()       # __device__
    HOST = auto()         # __host__
    SHARED = auto()       # __shared__
    CONSTANT = auto()     # __constant__
    RESTRICT = auto()     # __restrict__
    
    # C Keywords
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    STRUCT = auto()
    TYPEDEF = auto()
    EXTERN = auto()
    STATIC = auto()
    INLINE = auto()
    CONST = auto()
    VOLATILE = auto()
    SIZEOF = auto()
    
    # Types
    VOID = auto()
    INT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    CHAR_TYPE = auto()
    SHORT = auto()
    LONG = auto()
    UNSIGNED = auto()
    SIGNED = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AMPERSAND = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    EXCLAIM = auto()
    EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    QUESTION = auto()
    COLON = auto()
    DOT = auto()
    ARROW = auto()
    
    # Compound operators
    PLUS_PLUS = auto()
    MINUS_MINUS = auto()
    PLUS_EQUAL = auto()
    MINUS_EQUAL = auto()
    STAR_EQUAL = auto()
    SLASH_EQUAL = auto()
    EQUAL_EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    AND_AND = auto()
    OR_OR = auto()
    LESS_LESS = auto()
    GREATER_GREATER = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    
    # Special
    TRIPLE_CHEVRON_OPEN = auto()   # <<<
    TRIPLE_CHEVRON_CLOSE = auto()  # >>>
    PREPROCESSOR = auto()
    COMMENT = auto()
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    """Represents a lexical token."""
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# =============================================================================
# Lexer
# =============================================================================

class CUDALexer:
    """
    Lexical analyzer for CUDA source code.
    
    Handles CUDA-specific tokens like __global__, __shared__, and <<<>>>.
    """
    
    CUDA_KEYWORDS = {
        '__global__': TokenType.GLOBAL,
        '__device__': TokenType.DEVICE,
        '__host__': TokenType.HOST,
        '__shared__': TokenType.SHARED,
        '__constant__': TokenType.CONSTANT,
        '__restrict__': TokenType.RESTRICT,
    }
    
    C_KEYWORDS = {
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'do': TokenType.DO,
        'return': TokenType.RETURN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'struct': TokenType.STRUCT,
        'typedef': TokenType.TYPEDEF,
        'extern': TokenType.EXTERN,
        'static': TokenType.STATIC,
        'inline': TokenType.INLINE,
        'const': TokenType.CONST,
        'volatile': TokenType.VOLATILE,
        'sizeof': TokenType.SIZEOF,
        'void': TokenType.VOID,
        'int': TokenType.INT,
        'float': TokenType.FLOAT,
        'double': TokenType.DOUBLE,
        'char': TokenType.CHAR_TYPE,
        'short': TokenType.SHORT,
        'long': TokenType.LONG,
        'unsigned': TokenType.UNSIGNED,
        'signed': TokenType.SIGNED,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset: int = 1) -> Optional[str]:
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> str:
        char = self.current_char()
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_line_comment(self):
        # Skip //
        self.advance()
        self.advance()
        start_line = self.line
        start_col = self.column - 2
        comment = "//"
        while self.current_char() and self.current_char() != '\n':
            comment += self.advance()
        self.tokens.append(Token(TokenType.COMMENT, comment, start_line, start_col))
    
    def skip_block_comment(self):
        # Skip /*
        self.advance()
        self.advance()
        start_line = self.line
        start_col = self.column - 2
        comment = "/*"
        while self.current_char():
            if self.current_char() == '*' and self.peek() == '/':
                comment += self.advance()
                comment += self.advance()
                break
            comment += self.advance()
        self.tokens.append(Token(TokenType.COMMENT, comment, start_line, start_col))
    
    def read_string(self, quote: str) -> Token:
        start_line = self.line
        start_col = self.column
        value = self.advance()  # Opening quote
        
        while self.current_char() and self.current_char() != quote:
            if self.current_char() == '\\':
                value += self.advance()
                if self.current_char():
                    value += self.advance()
            else:
                value += self.advance()
        
        if self.current_char() == quote:
            value += self.advance()  # Closing quote
        
        token_type = TokenType.STRING if quote == '"' else TokenType.CHAR
        return Token(token_type, value, start_line, start_col)
    
    def read_number(self) -> Token:
        start_line = self.line
        start_col = self.column
        value = ""

        # Handle hex, octal, binary
        if self.current_char() == '0' and self.peek() is not None and self.peek() in 'xXoObB':
            value += self.advance()
            value += self.advance()
        
        while self.current_char() and (
            self.current_char().isalnum() or self.current_char() in '.eEpP+-'
        ):
            value += self.advance()
        
        # Handle suffixes
        while self.current_char() and self.current_char() in 'uUlLfF':
            value += self.advance()
        
        return Token(TokenType.NUMBER, value, start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.column
        value = ""
        
        while self.current_char() and (
            self.current_char().isalnum() or self.current_char() == '_'
        ):
            value += self.advance()
        
        # Check for keywords
        if value in self.CUDA_KEYWORDS:
            return Token(self.CUDA_KEYWORDS[value], value, start_line, start_col)
        if value in self.C_KEYWORDS:
            return Token(self.C_KEYWORDS[value], value, start_line, start_col)
        
        return Token(TokenType.IDENTIFIER, value, start_line, start_col)
    
    def read_preprocessor(self) -> Token:
        start_line = self.line
        start_col = self.column
        value = self.advance()  # #
        
        while self.current_char() and self.current_char() != '\n':
            if self.current_char() == '\\' and self.peek() == '\n':
                value += self.advance()
                value += self.advance()
            else:
                value += self.advance()
        
        return Token(TokenType.PREPROCESSOR, value, start_line, start_col)
    
    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        self.tokens = []
        
        while self.current_char():
            # Skip whitespace
            if self.current_char() in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Newlines
            if self.current_char() == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue
            
            # Comments
            if self.current_char() == '/' and self.peek() == '/':
                self.skip_line_comment()
                continue
            if self.current_char() == '/' and self.peek() == '*':
                self.skip_block_comment()
                continue
            
            # Preprocessor
            if self.current_char() == '#':
                self.tokens.append(self.read_preprocessor())
                continue
            
            # Strings and chars
            if self.current_char() == '"':
                self.tokens.append(self.read_string('"'))
                continue
            if self.current_char() == "'":
                self.tokens.append(self.read_string("'"))
                continue
            
            # Numbers
            if self.current_char().isdigit():
                self.tokens.append(self.read_number())
                continue
            if self.current_char() == '.' and self.peek() and self.peek().isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # CUDA kernel launch <<<>>>
            if (self.current_char() == '<' and 
                self.peek(1) == '<' and 
                self.peek(2) == '<'):
                line, col = self.line, self.column
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.TRIPLE_CHEVRON_OPEN, '<<<', line, col))
                continue
            
            if (self.current_char() == '>' and 
                self.peek(1) == '>' and 
                self.peek(2) == '>'):
                line, col = self.line, self.column
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.TRIPLE_CHEVRON_CLOSE, '>>>', line, col))
                continue
            
            # Multi-character operators
            line, col = self.line, self.column
            char = self.current_char()
            next_char = self.peek()
            
            two_char = char + (next_char or '')
            
            two_char_ops = {
                '++': TokenType.PLUS_PLUS,
                '--': TokenType.MINUS_MINUS,
                '+=': TokenType.PLUS_EQUAL,
                '-=': TokenType.MINUS_EQUAL,
                '*=': TokenType.STAR_EQUAL,
                '/=': TokenType.SLASH_EQUAL,
                '==': TokenType.EQUAL_EQUAL,
                '!=': TokenType.NOT_EQUAL,
                '<=': TokenType.LESS_EQUAL,
                '>=': TokenType.GREATER_EQUAL,
                '&&': TokenType.AND_AND,
                '||': TokenType.OR_OR,
                '<<': TokenType.LESS_LESS,
                '>>': TokenType.GREATER_GREATER,
                '->': TokenType.ARROW,
            }
            
            if two_char in two_char_ops:
                self.advance()
                self.advance()
                self.tokens.append(Token(two_char_ops[two_char], two_char, line, col))
                continue
            
            # Single-character operators
            single_char_ops = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '%': TokenType.PERCENT,
                '&': TokenType.AMPERSAND,
                '|': TokenType.PIPE,
                '^': TokenType.CARET,
                '~': TokenType.TILDE,
                '!': TokenType.EXCLAIM,
                '=': TokenType.EQUAL,
                '<': TokenType.LESS,
                '>': TokenType.GREATER,
                '?': TokenType.QUESTION,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
            }
            
            if char in single_char_ops:
                self.advance()
                self.tokens.append(Token(single_char_ops[char], char, line, col))
                continue
            
            # Unknown character - skip with warning
            self.advance()
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


# =============================================================================
# Source-to-Source Transformer
# =============================================================================

class CUDAToRIPPLETransformer:
    """
    Transforms CUDA source code to RIPPLE C code.
    
    This is the main entry point for source-level translation.
    """
    
    def __init__(self, ctx: Optional[TranslationContext] = None):
        self.ctx = ctx or TranslationContext(target_platform="hexagon")
        self.rule_engine = TranslationRuleEngine()
        self.hexagon_config = HexagonConfig()
    
    def transform(self, cuda_source: str) -> str:
        """
        Transform CUDA source to RIPPLE C.
        
        Returns the transformed source code.
        """
        # AST pre-pass: structural validation and kernel detection.
        # This does not drive code generation — it exists to catch malformed
        # input and enumerate kernels structurally (by parsing signatures,
        # not regex-scanning raw text) before the regex pass runs, and to
        # surface a warning instead of silently mistranslating when the
        # parser can't make sense of the source.
        try:
            lexer = CUDALexer(cuda_source)
            tokens = lexer.tokenize()
            builder = AIRBuilder(tokens, self.ctx)
            translation_unit = builder.build_translation_unit()
            detected_kernels = [f.name for f in translation_unit.functions if f.is_kernel]
            if not detected_kernels:
                self.ctx.add_warning(
                    "AST pre-pass did not recognize any __global__ kernel signatures in the "
                    "source (the AST parser's grammar covers common patterns, not the full "
                    "CUDA language, so this can be a false negative rather than a real "
                    "absence); translation will proceed via regex rules regardless."
                )
        except Exception as e:
            self.ctx.add_warning(f"AST pre-pass failed with {type(e).__name__}, proceeding with regex-only translation: {e}")

        # -- Regex Transformation (current translation path; AST does not drive codegen yet) --
        
        # Phase 1: Preprocess - extract structure
        preprocessed = self._preprocess(cuda_source)
        
        # Phase 2: Apply translation rules
        transformed = self.rule_engine.apply_all(preprocessed, self.ctx)
        
        # Phase 3: Add RIPPLE boilerplate
        result = self._add_ripple_boilerplate(transformed, cuda_source)
        
        # Phase 4: Post-process and format
        result = self._postprocess(result)
        
        return result
    
    def _preprocess(self, source: str) -> str:
        """Preprocess CUDA source - handle includes, extract kernels."""
        # Remove CUDA-specific includes
        result = re.sub(
            r'#include\s*<cuda_runtime\.h>',
            '#include <ripple.h>',
            source
        )
        result = re.sub(
            r'#include\s*<cuda\.h>',
            '#include <ripple.h>',
            result
        )
        result = re.sub(
            r'#include\s*<cooperative_groups\.h>',
            '// #include <cooperative_groups.h>  // Not needed in RIPPLE',
            result
        )
        
        return result
    
    def _add_ripple_boilerplate(self, source: str, original: str) -> str:
        """Add RIPPLE-specific boilerplate code."""
        # Infer block shape from original CUDA code
        block_shape = infer_block_shape(original, ctx=self.ctx)
        
        header = f"""/*
 * Auto-generated RIPPLE code from CUDA source
 * Target: Hexagon HVX ({self.hexagon_config.hvx_mode})
 * Vector width: {self.hexagon_config.hvx_width} bytes
 * 
 * Translation warnings:
"""
        for warning in self.ctx.warnings:
            header += f" *   - {warning}\n"
        
        header += f""" */

#include <ripple.h>
#include <stdint.h>
#include <stddef.h>

/* Hexagon-specific includes */
#ifdef __HEXAGON__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

/* Block shape configuration */
#define HVX_VECTOR_SIZE {self.hexagon_config.hvx_width}
#define RIPPLE_BLOCK_DIM_X {block_shape.dimensions[0]}
#define RIPPLE_BLOCK_DIM_Y {block_shape.dimensions[1] if len(block_shape.dimensions) > 1 else 1}
#define RIPPLE_BLOCK_DIM_Z {block_shape.dimensions[2] if len(block_shape.dimensions) > 2 else 1}

"""
        
        # Add helper macros
        header += """
/* RIPPLE block initialization */
#define RIPPLE_SETUP_BLOCK() \\
    ripple_block_t ripple_block = ripple_set_block_shape(HVX_PE, RIPPLE_BLOCK_DIM_X, RIPPLE_BLOCK_DIM_Y, RIPPLE_BLOCK_DIM_Z)

/* Atomic operation wrappers for Hexagon */
#ifdef __HEXAGON__
#define ripple_atomic_add(ptr, val) __builtin_atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define ripple_atomic_max(ptr, val) __sync_fetch_and_max(ptr, val)
#define ripple_atomic_min(ptr, val) __sync_fetch_and_min(ptr, val)
#define ripple_atomic_cas(ptr, cmp, val) __sync_val_compare_and_swap(ptr, cmp, val)
#define ripple_atomic_exch(ptr, val) __sync_lock_test_and_set(ptr, val)
#else
#define ripple_atomic_add(ptr, val) (*(ptr) += (val))
#define ripple_atomic_max(ptr, val) do { if ((val) > *(ptr)) *(ptr) = (val); } while(0)
#define ripple_atomic_min(ptr, val) do { if ((val) < *(ptr)) *(ptr) = (val); } while(0)
#define ripple_atomic_cas(ptr, cmp, val) (*(ptr) == (cmp) ? (*(ptr) = (val), (cmp)) : *(ptr))
#define ripple_atomic_exch(ptr, val) ({ typeof(*(ptr)) __old = *(ptr); *(ptr) = (val); __old; })
#endif

/* Math intrinsics */
#define ripple_sad(x, y, z) (__builtin_abs((x) - (y)) + (z))

"""

        # Hoisted declarations (e.g. warp-shuffle permutation functions —
        # C has no closures, so these are named, file-scope functions
        # generated by rules like ShuffleXorRule). Placed after the
        # standard header and before any translated kernel body in
        # `source`, so "must be defined before use" holds for every
        # kernel in the file regardless of which one references which
        # hoisted function.
        if self.ctx.hoisted_declarations:
            header += "/* Shuffle permutation functions (hoisted to file scope) */\n"
            header += "\n".join(self.ctx.hoisted_declarations)
            header += "\n\n"

        return header + source
    
    def _postprocess(self, source: str) -> str:
        """Post-process the transformed source."""
        # Clean up multiple blank lines
        source = re.sub(r'\n{3,}', '\n\n', source)
        
        # Ensure proper indentation (basic)
        lines = source.split('\n')
        result = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Decrease indent before closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add line with proper indent
            if stripped:
                result.append('    ' * indent_level + stripped)
            else:
                result.append('')
            
            # Increase indent after opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(result)
    
    def transform_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Transform a CUDA file to RIPPLE."""
        with open(input_path, 'r') as f:
            cuda_source = f.read()
        
        ripple_source = self.transform(cuda_source)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(ripple_source)
        
        return ripple_source


# =============================================================================
# AIR Builder (for structured transformation)
# =============================================================================

class AIRBuilder:
    """
    Builds AIR from tokenized CUDA source.
    
    This provides a more structured approach for complex transformations.
    """
    
    def __init__(self, tokens: list[Token], ctx: TranslationContext):
        self.tokens = [t for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.COMMENT)]
        self.pos = 0
        self.ctx = ctx
    
    def current(self) -> Token:
        if self.pos >= len(self.tokens):
            return Token(TokenType.EOF, '', 0, 0)
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 1) -> Token:
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return Token(TokenType.EOF, '', 0, 0)
        return self.tokens[pos]
    
    def advance(self) -> Token:
        token = self.current()
        self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name} but got {token.type.name} "
                f"at line {token.line}:{token.column}"
            )
        return self.advance()
    
    def build_translation_unit(self) -> AIRTranslationUnit:
        """Build a complete translation unit."""
        unit = AIRTranslationUnit(target_platform="hexagon")
        
        while self.current().type != TokenType.EOF:
            if self.current().type == TokenType.PREPROCESSOR:
                # Handle preprocessor directives
                directive = self.advance().value
                if directive.startswith('#include'):
                    unit.includes.append(directive)
                continue
            
            # Check for function/kernel definition
            if self._is_function_start():
                pos_before = self.pos
                func = self._parse_function()
                if func:
                    unit.functions.append(func)
                if self.pos == pos_before:
                    # _is_function_start() matched a qualifier (e.g. STATIC
                    # or INLINE) that _parse_type() doesn't know how to
                    # consume on its own, so _parse_function() bailed via
                    # return None without advancing at all (e.g. a bare
                    # top-level "static int counter;" declaration). Skip
                    # the token rather than retrying it forever.
                    self.advance()
            else:
                self.advance()  # Skip unknown tokens
        
        return unit
    
    def _is_function_start(self) -> bool:
        """Check if current position starts a function definition."""
        return self.current().type in (
            TokenType.GLOBAL, TokenType.DEVICE, TokenType.HOST,
            TokenType.VOID, TokenType.INT, TokenType.FLOAT,
            TokenType.DOUBLE, TokenType.STATIC, TokenType.INLINE
        )
    
    def _parse_function(self) -> Optional[AIRFunction]:
        """Parse a function or kernel definition."""
        # Collect attributes
        is_kernel = False
        is_device = False
        
        while self.current().type in (TokenType.GLOBAL, TokenType.DEVICE, TokenType.HOST):
            if self.current().type == TokenType.GLOBAL:
                is_kernel = True
            elif self.current().type == TokenType.DEVICE:
                is_device = True
            self.advance()

        # Skip __launch_bounds__(...) and similar call-like attributes that
        # can appear between the __global__/__device__ qualifiers and the
        # return type (e.g. `__global__ __launch_bounds__(256) void k(...)`).
        # Recognized by name rather than a generic "any identifier(...)"
        # heuristic, to avoid accidentally swallowing a real return-type-like
        # macro.
        while (self.current().type == TokenType.IDENTIFIER
               and self.current().value == '__launch_bounds__'):
            self.advance()  # consume __launch_bounds__
            if self.current().type == TokenType.LPAREN:
                self.advance()
                paren_depth = 1
                while paren_depth > 0 and self.current().type != TokenType.EOF:
                    if self.current().type == TokenType.LPAREN:
                        paren_depth += 1
                    elif self.current().type == TokenType.RPAREN:
                        paren_depth -= 1
                    self.advance()

        # Parse return type
        return_type = self._parse_type()
        
        # Parse function name
        if self.current().type != TokenType.IDENTIFIER:
            return None
        name = self.advance().value
        
        # Parse parameters
        if self.current().type != TokenType.LPAREN:
            return None
        self.advance()  # Skip (
        
        parameters = []
        while self.current().type != TokenType.RPAREN:
            if self.current().type == TokenType.EOF:
                return None
            pos_before = self.pos
            param = self._parse_parameter()
            if param:
                parameters.append(param)
            if self.current().type == TokenType.COMMA:
                self.advance()
            if self.pos == pos_before:
                # _parse_parameter couldn't interpret the current token
                # (e.g. a bare NUMBER or an unhandled punctuation token
                # like '&' in parameter position) and made no progress —
                # bail rather than spin forever re-parsing the same token.
                return None
        
        self.expect(TokenType.RPAREN)
        
        # Parse body (simplified - just collect tokens until matching })
        if self.current().type != TokenType.LBRACE:
            return None
        
        body = self._parse_block()
        
        return AIRFunction(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            is_kernel=is_kernel,
            is_device=is_device
        )
    
    def _parse_type(self) -> AIRType:
        """Parse a type specification."""
        is_const = False
        is_unsigned = False
        base_type = ""
        
        while self.current().type in (TokenType.CONST, TokenType.VOLATILE,
                                       TokenType.UNSIGNED, TokenType.SIGNED):
            if self.current().type == TokenType.CONST:
                is_const = True
            elif self.current().type == TokenType.UNSIGNED:
                is_unsigned = True
            self.advance()
        
        # Base type
        if self.current().type in (TokenType.VOID, TokenType.INT, TokenType.FLOAT,
                                    TokenType.DOUBLE, TokenType.CHAR_TYPE,
                                    TokenType.SHORT, TokenType.LONG):
            base_type = self.advance().value
        elif self.current().type == TokenType.IDENTIFIER:
            base_type = self.advance().value
        
        if is_unsigned:
            base_type = "unsigned " + base_type
        
        # Pointer
        pointer_depth = 0
        while self.current().type == TokenType.STAR:
            pointer_depth += 1
            self.advance()
        
        return AIRType(
            base_type=base_type,
            is_pointer=pointer_depth > 0,
            pointer_depth=pointer_depth,
            is_const=is_const
        )
    
    def _parse_parameter(self) -> Optional[AIRVariable]:
        """Parse a function parameter."""
        param_type = self._parse_type()
        
        if self.current().type == TokenType.IDENTIFIER:
            name = self.advance().value
        else:
            name = f"param_{self.pos}"
        
        # Handle array syntax
        while self.current().type == TokenType.LBRACKET:
            self.advance()
            while self.current().type != TokenType.RBRACKET:
                if self.current().type == TokenType.EOF:
                    break
                self.advance()
            if self.current().type == TokenType.RBRACKET:
                self.advance()
            param_type.is_pointer = True
            param_type.pointer_depth += 1
        
        return AIRVariable(
            name=name,
            var_type=param_type,
            is_parameter=True
        )
    
    def _parse_block(self) -> list[AIRNode]:
        """Parse a block of statements."""
        body = []

        self.expect(TokenType.LBRACE)
        brace_count = 1

        while brace_count > 0:
            if self.current().type == TokenType.LBRACE:
                brace_count += 1
                self.advance()
                continue
            elif self.current().type == TokenType.RBRACE:
                brace_count -= 1
                self.advance()
                continue
            elif self.current().type == TokenType.EOF:
                break

            # Collect statement as expression
            stmt_tokens = []
            while self.current().type not in (TokenType.SEMICOLON, TokenType.LBRACE,
                                               TokenType.RBRACE, TokenType.EOF):
                stmt_tokens.append(self.advance())

            if stmt_tokens:
                expr = ' '.join(t.value for t in stmt_tokens)
                body.append(AIRExpression(expr=expr))

            if self.current().type == TokenType.SEMICOLON:
                self.advance()

        return body


# =============================================================================
# Convenience Functions
# =============================================================================

def translate_cuda_source(cuda_source: str, target: str = "hexagon") -> str:
    """
    Translate CUDA source code to RIPPLE.
    
    Args:
        cuda_source: CUDA source code string
        target: Target platform ("hexagon", "x86", "arm")
    
    Returns:
        RIPPLE C source code string
    """
    ctx = TranslationContext(target_platform=target)
    transformer = CUDAToRIPPLETransformer(ctx)
    return transformer.transform(cuda_source)


def translate_cuda_file(input_path: str, output_path: str, target: str = "hexagon") -> str:
    """
    Translate a CUDA source file to RIPPLE.
    
    Args:
        input_path: Path to CUDA source file
        output_path: Path to write RIPPLE output
        target: Target platform
    
    Returns:
        RIPPLE C source code string
    """
    ctx = TranslationContext(target_platform=target)
    transformer = CUDAToRIPPLETransformer(ctx)
    return transformer.transform_file(input_path, output_path)
