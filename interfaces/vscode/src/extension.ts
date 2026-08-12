/**
 * CUDA to RIPPLE VS Code Extension
 * 
 * Provides in-editor translation from CUDA to RIPPLE for Hexagon HVX targets.
 */

import * as vscode from 'vscode';
import * as path from 'path';

// =============================================================================
// Configuration
// =============================================================================

interface Cuda2RippleConfig {
    target: 'hexagon' | 'x86' | 'arm';
    hvxWidth: 64 | 128;
    hvxMode: string;
    preserveComments: boolean;
    showWarnings: boolean;
    translationMode: 'source' | 'ir';
    serverUrl: string;
}

function getConfig(): Cuda2RippleConfig {
    const config = vscode.workspace.getConfiguration('cuda2ripple');
    return {
        target: config.get('target', 'hexagon'),
        hvxWidth: config.get('hvxWidth', 128),
        hvxMode: config.get('hvxMode', 'v68'),
        preserveComments: config.get('preserveComments', true),
        showWarnings: config.get('showWarnings', true),
        translationMode: config.get('translationMode', 'source'),
        serverUrl: config.get('serverUrl', 'http://localhost:5000')
    };
}

// =============================================================================
// Translation Engine (Local)
// =============================================================================

// NOTE: duplicates core/translation_rules.py's replacement rules
// independently (not a wrapper around the Python translator) — a rule
// fixed there must be fixed here too, or it silently drifts. See
// GitHub issue #9 for the bug this caused and why a full fix (having
// this extension call into the Python translator instead) was
// deliberately deferred as a separate, bigger decision.
class LocalTranslator {
    private config: Cuda2RippleConfig;

    constructor(config: Cuda2RippleConfig) {
        this.config = config;
    }

    translate(cudaCode: string): { output: string; warnings: string[] } {
        let output = cudaCode;
        const warnings: string[] = [];

        // Thread index
        output = output.replace(/threadIdx\.x/g, 'ripple_id(ripple_block, 0)');
        output = output.replace(/threadIdx\.y/g, 'ripple_id(ripple_block, 1)');
        output = output.replace(/threadIdx\.z/g, 'ripple_id(ripple_block, 2)');

        // Block dimensions
        output = output.replace(/blockDim\.x/g, 'ripple_get_block_size(ripple_block, 0)');
        output = output.replace(/blockDim\.y/g, 'ripple_get_block_size(ripple_block, 1)');
        output = output.replace(/blockDim\.z/g, 'ripple_get_block_size(ripple_block, 2)');

        // Block/grid indices
        output = output.replace(/blockIdx\.x/g, 'block_idx_x');
        output = output.replace(/blockIdx\.y/g, 'block_idx_y');
        output = output.replace(/gridDim\.x/g, 'grid_dim_x');
        output = output.replace(/gridDim\.y/g, 'grid_dim_y');

        // Kernel declaration
        output = output.replace(
            /__global__\s+void\s+(\w+)\s*\(([^)]*)\)/g,
            'void $1_ripple(\n    int grid_dim_x, int grid_dim_y, int grid_dim_z,\n    int block_dim_x, int block_dim_y, int block_dim_z,\n    $2)'
        );

        // Device function
        output = output.replace(/__device__\s+/g, 'static inline ');

        // Shared memory
        output = output.replace(
            /__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]*)\]/g,
            '__attribute__((section(".vtcm"))) __attribute__((aligned(128))) $1 $2[$3]'
        );

        // Sync
        output = output.replace(/__syncthreads\s*\(\s*\)/g, '/* __syncthreads: implicit in SIMD */');

        // Atomics
        output = output.replace(/atomicAdd\s*\(/g, 'ripple_atomic_add(');

        // Warnings
        if (cudaCode.includes('cooperative_groups')) {
            warnings.push('Cooperative groups require manual translation');
        }

        return { output: this.generateHeader() + output, warnings };
    }

    private generateHeader(): string {
        return `/*
 * Auto-generated RIPPLE code - Target: Hexagon HVX (${this.config.hvxMode})
 */
#include <ripple.h>
#include <stdint.h>

#define HVX_VECTOR_SIZE ${this.config.hvxWidth}
#define ripple_atomic_add(ptr, val) __builtin_atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)

`;
    }
}

// =============================================================================
// Commands
// =============================================================================

async function translateSelection(editor: vscode.TextEditor) {
    const config = getConfig();
    const selection = editor.selection;
    const text = editor.document.getText(selection);

    if (!text) {
        vscode.window.showWarningMessage('No text selected');
        return;
    }

    const translator = new LocalTranslator(config);
    const result = translator.translate(text);

    if (result.warnings.length > 0) {
        vscode.window.showWarningMessage(result.warnings.join('\n'));
    }

    await editor.edit(editBuilder => {
        editBuilder.replace(selection, result.output);
    });
}

async function translateFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const config = getConfig();
    const text = editor.document.getText();
    const translator = new LocalTranslator(config);
    const result = translator.translate(text);

    const newDoc = await vscode.workspace.openTextDocument({
        content: result.output,
        language: 'c'
    });
    await vscode.window.showTextDocument(newDoc, vscode.ViewColumn.Beside);
}

// =============================================================================
// Activation
// =============================================================================

export function activate(context: vscode.ExtensionContext) {
    console.log('CUDA to RIPPLE extension activated');

    context.subscriptions.push(
        vscode.commands.registerTextEditorCommand('cuda2ripple.translate', translateSelection),
        vscode.commands.registerCommand('cuda2ripple.translateFile', translateFile)
    );

    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.text = '$(zap) CUDA→RIPPLE';
    statusBar.command = 'cuda2ripple.translate';
    context.subscriptions.push(statusBar);

    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor && /\.(cu|cuda|cuh)$/.test(editor.document.fileName)) {
                statusBar.show();
            } else {
                statusBar.hide();
            }
        })
    );
}

export function deactivate() {}
