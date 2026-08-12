#!/usr/bin/env python3
"""
CUDA to RIPPLE Translator - Command Line Interface

Usage:
    cuda2ripple source input.cu -o output.c     # Source-level translation
    cuda2ripple ir input.ll -o output.ll        # IR-level translation
    cuda2ripple analyze input.cu                # Analyze without translation
    cuda2ripple batch *.cu -o outdir/           # Batch translation
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.semantic_model import TranslationContext, HexagonConfig
from core.translation_rules import TranslationRuleEngine
from frontends.source.cuda_frontend import (
    CUDAToRIPPLETransformer,
    translate_cuda_source,
    translate_cuda_file,
    CUDALexer
)
from frontends.ir.ir_frontend import (
    CUDAIRToRIPPLETranslator,
    translate_cuda_ir,
    analyze_cuda_ir
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = '', end='\n'):
    """Print with optional color."""
    if sys.stdout.isatty():
        print(f"{color}{text}{Colors.ENDC}", end=end)
    else:
        print(text, end=end)


def print_banner():
    """Print the application banner."""
    banner = r"""
   _____ _   _ ____    _       ____  _             _      
  / ____| | | |  _ \  / \     |  _ \(_)_ __  _ __ | | ___ 
 | |    | | | | | | |/ _ \    | |_) | | '_ \| '_ \| |/ _ \
 | |    | |_| | |_| / ___ \   |  _ <| | |_) | |_) | |  __/
  \____| \___/|____/_/   \_\  |_| \_\_| .__/| .__/|_|\___|
                                      |_|   |_|           
    """
    print_colored(banner, Colors.CYAN)
    print_colored("  CUDA to RIPPLE Translator for Hexagon HVX", Colors.BOLD)
    print_colored("  Version 0.1.0\n", Colors.BLUE)


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_source(args):
    """Handle source-level translation."""
    print_colored(f"[Source Translation] {args.input}", Colors.GREEN)
    
    # Create context
    ctx = TranslationContext(
        target_platform=args.target,
        translation_mode="source",
        preserve_comments=not args.strip_comments,
        generate_debug_info=args.debug
    )
    
    # Configure Hexagon
    if args.target == "hexagon":
        hexagon_config = HexagonConfig(
            hvx_width=args.hvx_width,
            hvx_mode=args.hvx_mode
        )
    
    try:
        # Read input
        with open(args.input, 'r') as f:
            cuda_source = f.read()
        
        # Transform
        transformer = CUDAToRIPPLETransformer(ctx)
        ripple_source = transformer.transform(cuda_source)
        
        # Output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(ripple_source)
            print_colored(f"[OK] Written to {args.output}", Colors.GREEN)
        else:
            print(ripple_source)
        
        # Print warnings
        if ctx.warnings:
            print_colored("\n[Warnings]", Colors.WARNING)
            for w in ctx.warnings:
                print_colored(f"  - {w}", Colors.WARNING)
        
        return 0
        
    except FileNotFoundError:
        print_colored(f"[Error] File not found: {args.input}", Colors.FAIL)
        return 1
    except Exception as e:
        print_colored(f"[Error] {e}", Colors.FAIL)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_ir(args):
    """Handle IR-level translation."""
    print_colored(f"[IR Translation] {args.input}", Colors.GREEN)
    
    ctx = TranslationContext(
        target_platform=args.target,
        translation_mode="ir"
    )
    
    try:
        # Read input
        with open(args.input, 'r') as f:
            cuda_ir = f.read()
        
        # Translate
        translator = CUDAIRToRIPPLETranslator(ctx)
        ripple_ir = translator.translate(cuda_ir)
        
        # Output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(ripple_ir)
            print_colored(f"[OK] Written to {args.output}", Colors.GREEN)
        else:
            print(ripple_ir)
        
        return 0
        
    except FileNotFoundError:
        print_colored(f"[Error] File not found: {args.input}", Colors.FAIL)
        return 1
    except Exception as e:
        print_colored(f"[Error] {e}", Colors.FAIL)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_analyze(args):
    """Analyze CUDA code without translation."""
    print_colored(f"[Analyzing] {args.input}", Colors.CYAN)
    
    try:
        with open(args.input, 'r') as f:
            source = f.read()
        
        # Determine if source or IR
        is_ir = args.input.endswith('.ll') or args.input.endswith('.bc')
        
        if is_ir:
            analysis = analyze_cuda_ir(source)
            result = {
                "type": "IR",
                "file": args.input,
                "kernels": len(analysis.kernels),
                "device_functions": len(analysis.device_functions),
                "max_block_dim": analysis.max_block_dim,
                "uses_shared_memory": analysis.uses_shared_memory,
                "uses_warp_shuffles": analysis.uses_warp_shuffles,
                "uses_atomics": analysis.uses_atomics,
                "thread_idx_uses": len(analysis.thread_idx_uses),
                "shuffle_calls": len(analysis.shuffle_calls),
                "atomic_calls": len(analysis.atomic_calls)
            }
        else:
            # Analyze source
            lexer = CUDALexer(source)
            tokens = lexer.tokenize()
            
            # Count patterns
            result = {
                "type": "Source",
                "file": args.input,
                "lines": len(source.split('\n')),
                "tokens": len(tokens),
                "has_global_kernels": '__global__' in source,
                "has_device_functions": '__device__' in source,
                "has_shared_memory": '__shared__' in source,
                "uses_threadIdx": 'threadIdx' in source,
                "uses_blockIdx": 'blockIdx' in source,
                "uses_syncthreads": '__syncthreads' in source,
                "uses_shuffles": '__shfl' in source,
                "uses_atomics": 'atomic' in source,
            }
        
        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_colored("\n[Analysis Results]", Colors.BOLD)
            for key, value in result.items():
                print_colored(f"  {key}: ", Colors.CYAN, end='')
                print(value)
        
        return 0
        
    except Exception as e:
        print_colored(f"[Error] {e}", Colors.FAIL)
        return 1


def cmd_batch(args):
    """Batch translate multiple files."""
    print_colored(f"[Batch Translation] {len(args.inputs)} files", Colors.GREEN)
    
    # Create output directory
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0

    for input_path in args.inputs:
        try:
            input_path = Path(input_path)
            output_path = outdir / input_path.with_suffix('.ripple.c').name

            print_colored(f"  {input_path.name} -> {output_path.name}... ", end='')

            with open(input_path, 'r') as f:
                cuda_source = f.read()

            # Fresh context per file: TranslationContext accumulates
            # per-translation state (errors, warnings, hoisted_declarations
            # — see core/semantic_model.py) that must not leak between
            # unrelated files translated in the same batch run. Reusing a
            # loop-level ctx/transformer here previously caused warnings and
            # hoisted declarations to bleed across files, and once any rule
            # calls ctx.add_error(), it would also cause ctx.has_errors() to
            # stay True for every file after the first failure, misreporting
            # otherwise-successful later files as FAILED. Same bug class
            # cmd_interactive's 'file' handler already fixed — see its
            # comment below.
            ctx = TranslationContext(target_platform=args.target)
            transformer = CUDAToRIPPLETransformer(ctx)
            ripple_source = transformer.transform(cuda_source)
            
            with open(output_path, 'w') as f:
                f.write(ripple_source)
            
            print_colored("OK", Colors.GREEN)
            success += 1
            
        except Exception as e:
            print_colored(f"FAILED: {e}", Colors.FAIL)
            failed += 1
    
    print_colored(f"\n[Complete] {success} succeeded, {failed} failed", 
                  Colors.GREEN if failed == 0 else Colors.WARNING)
    
    return 0 if failed == 0 else 1


def cmd_interactive(args):
    """Interactive REPL mode."""
    print_banner()
    print_colored("Interactive mode. Type 'help' for commands, 'quit' to exit.\n", Colors.CYAN)
    
    ctx = TranslationContext(target_platform=args.target)
    rule_engine = TranslationRuleEngine()
    
    while True:
        try:
            line = input(f"{Colors.GREEN}cuda2ripple> {Colors.ENDC}")
            line = line.strip()
            
            if not line:
                continue
            
            if line in ('quit', 'exit', 'q'):
                print_colored("Goodbye!", Colors.CYAN)
                break
            
            if line == 'help':
                print("""
Commands:
  help              Show this help
  quit/exit/q       Exit interactive mode
  rules             List translation rules
  config            Show current configuration
  translate <code>  Translate inline CUDA code
  file <path>       Translate a file
                """)
                continue
            
            if line == 'rules':
                print_colored("\n[Translation Rules]", Colors.BOLD)
                for rule in rule_engine.rules:
                    print(f"  [{rule.priority:3d}] {rule.name}: {rule.description}")
                print()
                continue
            
            if line == 'config':
                print_colored("\n[Configuration]", Colors.BOLD)
                print(f"  Target: {ctx.target_platform}")
                print(f"  Mode: {ctx.translation_mode}")
                print(f"  Vector width: {HexagonConfig().hvx_width} bytes")
                print()
                continue
            
            if line.startswith('translate '):
                cuda_code = line[10:]
                ripple_code = rule_engine.apply_all(cuda_code, ctx)
                print_colored("\n[Translated]", Colors.BOLD)
                print(ripple_code)
                print()
                continue
            
            if line.startswith('file '):
                filepath = line[5:].strip()
                try:
                    with open(filepath, 'r') as f:
                        cuda_source = f.read()
                    # Fresh context per file: TranslationContext now
                    # accumulates per-translation state (hoisted_declarations,
                    # warnings — see core/semantic_model.py) that must not
                    # leak between unrelated files translated in the same
                    # REPL session. Reusing the loop-level ctx here caused a
                    # translated file with no shuffle calls to still show a
                    # previous file's unreferenced hoisted shuffle helper.
                    ctx = TranslationContext(target_platform=args.target)
                    transformer = CUDAToRIPPLETransformer(ctx)
                    ripple_source = transformer.transform(cuda_source)
                    print(ripple_source)
                except Exception as e:
                    print_colored(f"Error: {e}", Colors.FAIL)
                continue
            
            # Try to translate as inline code
            ripple_code = rule_engine.apply_all(line, ctx)
            if ripple_code != line:
                print_colored(f"  -> {ripple_code}", Colors.CYAN)
            else:
                print_colored("Unknown command. Type 'help' for commands.", Colors.WARNING)
                
        except KeyboardInterrupt:
            print("\n")
            continue
        except EOFError:
            print_colored("\nGoodbye!", Colors.CYAN)
            break


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CUDA to RIPPLE Translator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cuda2ripple source kernel.cu -o kernel.ripple.c
  cuda2ripple ir kernel.ll -o kernel.ripple.ll
  cuda2ripple analyze kernel.cu --json
  cuda2ripple batch *.cu -o output/
  cuda2ripple interactive
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--version', action='version', version='cuda2ripple 0.1.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Translation mode')
    
    # Source command
    source_parser = subparsers.add_parser('source', help='Source-level translation')
    source_parser.add_argument('input', help='Input CUDA source file')
    source_parser.add_argument('-o', '--output', help='Output file')
    source_parser.add_argument('--target', default='hexagon',
                               choices=['hexagon', 'x86', 'arm'],
                               help='Target platform')
    source_parser.add_argument('--hvx-width', type=int, default=128,
                               choices=[64, 128],
                               help='HVX vector width in bytes')
    source_parser.add_argument('--hvx-mode', default='v68',
                               help='HVX instruction set version')
    source_parser.add_argument('--strip-comments', action='store_true',
                               help='Strip comments from output')
    source_parser.add_argument('--debug', action='store_true',
                               help='Generate debug information')
    
    # IR command
    ir_parser = subparsers.add_parser('ir', help='IR-level translation')
    ir_parser.add_argument('input', help='Input LLVM IR file')
    ir_parser.add_argument('-o', '--output', help='Output file')
    ir_parser.add_argument('--target', default='hexagon',
                           choices=['hexagon', 'x86', 'arm'],
                           help='Target platform')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze CUDA code')
    analyze_parser.add_argument('input', help='Input file')
    analyze_parser.add_argument('--json', action='store_true',
                                help='Output as JSON')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch translation')
    batch_parser.add_argument('inputs', nargs='+', help='Input files')
    batch_parser.add_argument('-o', '--output', required=True,
                              help='Output directory')
    batch_parser.add_argument('--target', default='hexagon',
                              choices=['hexagon', 'x86', 'arm'],
                              help='Target platform')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', 
                                                help='Interactive REPL mode')
    interactive_parser.add_argument('--target', default='hexagon',
                                    choices=['hexagon', 'x86', 'arm'],
                                    help='Target platform')
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return 0
    
    if args.command == 'source':
        return cmd_source(args)
    elif args.command == 'ir':
        return cmd_ir(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'interactive':
        return cmd_interactive(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
