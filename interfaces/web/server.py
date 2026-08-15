"""
CUDA to RIPPLE Translator - Web Interface

A Flask-based web application for translating CUDA to RIPPLE.
Provides a live editor with syntax highlighting and side-by-side comparison.
"""

from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.semantic_model import TranslationContext, HexagonConfig
from frontends.source.cuda_frontend import CUDAToRIPPLETransformer, CUDALexer
from frontends.ir.ir_frontend import CUDAIRToRIPPLETranslator

app = Flask(__name__)

# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA to RIPPLE Translator</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-editor: #0f0f1a;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent-cuda: #76b900;
            --accent-ripple: #00b4d8;
            --border: #333;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo h1 {
            font-size: 1.5rem;
            background: linear-gradient(90deg, var(--accent-cuda), var(--accent-ripple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .logo-icon {
            font-size: 2rem;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        select, button {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        select:hover, button:hover {
            border-color: var(--accent-ripple);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-cuda), var(--accent-ripple));
            border: none;
            font-weight: bold;
        }
        
        .btn-primary:hover {
            opacity: 0.9;
        }
        
        .main {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: var(--border);
            height: calc(100vh - 120px);
        }
        
        .editor-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
        }
        
        .panel-header {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-title {
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .panel-title.cuda { color: var(--accent-cuda); }
        .panel-title.ripple { color: var(--accent-ripple); }
        
        .editor-container {
            flex: 1;
            overflow: hidden;
            position: relative;
        }
        
        textarea {
            width: 100%;
            height: 100%;
            background: var(--bg-editor);
            color: var(--text-primary);
            border: none;
            padding: 1rem;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.6;
            resize: none;
            outline: none;
        }
        
        .output-display {
            width: 100%;
            height: 100%;
            background: var(--bg-editor);
            color: var(--text-primary);
            padding: 1rem;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.6;
            overflow: auto;
            white-space: pre-wrap;
        }
        
        .status-bar {
            background: var(--bg-secondary);
            padding: 0.5rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-ripple);
        }
        
        .status-dot.warning { background: #ffc107; }
        .status-dot.error { background: #dc3545; }
        
        .warnings-panel {
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 0.75rem 1rem;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .warning-item {
            color: #ffc107;
            font-size: 0.85rem;
            padding: 0.25rem 0;
        }
        
        .warning-item::before {
            content: "⚠️ ";
        }
        
        /* Syntax highlighting */
        .keyword { color: #c586c0; }
        .type { color: #4ec9b0; }
        .function { color: #dcdcaa; }
        .string { color: #ce9178; }
        .comment { color: #6a9955; }
        .number { color: #b5cea8; }
        
        /* Loading spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid transparent;
            border-top-color: var(--accent-ripple);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .spinner.active { display: inline-block; }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Mode tabs */
        .mode-tabs {
            display: flex;
            gap: 0;
        }
        
        .mode-tab {
            padding: 0.5rem 1rem;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
        }
        
        .mode-tab:first-child {
            border-radius: 4px 0 0 4px;
        }
        
        .mode-tab:last-child {
            border-radius: 0 4px 4px 0;
        }
        
        .mode-tab.active {
            background: var(--accent-ripple);
            color: var(--text-primary);
            border-color: var(--accent-ripple);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <span class="logo-icon">🔄</span>
            <h1>CUDA → RIPPLE Translator</h1>
        </div>
        <div class="controls">
            <div class="mode-tabs">
                <button class="mode-tab active" data-mode="source">Source</button>
                <button class="mode-tab" data-mode="ir">IR</button>
            </div>
            <div class="control-group">
                <label>Target:</label>
                <select id="target">
                    <option value="hexagon" selected>Hexagon HVX</option>
                    <option value="x86">x86-64 AVX</option>
                    <option value="arm">ARM SVE</option>
                </select>
            </div>
            <div class="control-group">
                <label>HVX Width:</label>
                <select id="hvx-width">
                    <option value="128" selected>128 bytes</option>
                    <option value="64">64 bytes</option>
                </select>
            </div>
            <button class="btn-primary" id="translate-btn">
                <span class="spinner" id="spinner"></span>
                Translate
            </button>
        </div>
    </div>
    
    <div class="main">
        <div class="editor-panel">
            <div class="panel-header">
                <span class="panel-title cuda">📥 CUDA Input</span>
                <span id="input-stats">0 lines</span>
            </div>
            <div class="editor-container">
                <textarea id="cuda-input" placeholder="Paste your CUDA code here...">// Example CUDA kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Reduction kernel with shared memory
__global__ void reduce_sum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[0] = sdata[0];
    }
}</textarea>
            </div>
        </div>
        
        <div class="editor-panel">
            <div class="panel-header">
                <span class="panel-title ripple">📤 RIPPLE Output</span>
                <span id="output-stats">-</span>
            </div>
            <div class="editor-container">
                <div class="output-display" id="ripple-output">Click "Translate" to convert CUDA to RIPPLE...</div>
            </div>
            <div class="warnings-panel" id="warnings-panel" style="display: none;">
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-item">
            <span class="status-dot" id="status-dot"></span>
            <span id="status-text">Ready</span>
        </div>
        <div class="status-item">
            <span>Target: Hexagon HVX v68 | Vector: 128 bytes | RIPPLE v0.1 | Compile with: clang -fenable-ripple ...</span>
        </div>
    </div>

    <script>
        const cudaInput = document.getElementById('cuda-input');
        const rippleOutput = document.getElementById('ripple-output');
        const translateBtn = document.getElementById('translate-btn');
        const spinner = document.getElementById('spinner');
        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        const inputStats = document.getElementById('input-stats');
        const outputStats = document.getElementById('output-stats');
        const warningsPanel = document.getElementById('warnings-panel');
        const modeTabs = document.querySelectorAll('.mode-tab');
        
        let currentMode = 'source';
        
        // Mode switching
        modeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                modeTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMode = tab.dataset.mode;
            });
        });
        
        // Update input stats
        cudaInput.addEventListener('input', () => {
            const lines = cudaInput.value.split('\\n').length;
            inputStats.textContent = `${lines} lines`;
        });
        
        // Translate button
        translateBtn.addEventListener('click', async () => {
            const code = cudaInput.value;
            const target = document.getElementById('target').value;
            const hvxWidth = document.getElementById('hvx-width').value;
            
            // Show loading state
            spinner.classList.add('active');
            translateBtn.disabled = true;
            statusText.textContent = 'Translating...';
            statusDot.className = 'status-dot';
            
            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        code: code,
                        mode: currentMode,
                        target: target,
                        hvx_width: parseInt(hvxWidth)
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    rippleOutput.textContent = result.output;
                    outputStats.textContent = `${result.output.split('\\n').length} lines`;
                    statusText.textContent = `Translated successfully (${result.time_ms.toFixed(1)}ms)`;
                    statusDot.className = 'status-dot';
                    
                    // Show warnings
                    if (result.warnings && result.warnings.length > 0) {
                        warningsPanel.style.display = 'block';
                        warningsPanel.innerHTML = result.warnings
                            .map(w => `<div class="warning-item">${w}</div>`)
                            .join('');
                        statusDot.classList.add('warning');
                    } else {
                        warningsPanel.style.display = 'none';
                    }
                } else {
                    rippleOutput.textContent = `Error: ${result.error}`;
                    statusText.textContent = 'Translation failed';
                    statusDot.classList.add('error');
                }
            } catch (error) {
                rippleOutput.textContent = `Error: ${error.message}`;
                statusText.textContent = 'Connection error';
                statusDot.classList.add('error');
            } finally {
                spinner.classList.remove('active');
                translateBtn.disabled = false;
            }
        });
        
        // Initial stats
        cudaInput.dispatchEvent(new Event('input'));
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                translateBtn.click();
            }
        });
    </script>
</body>
</html>
'''


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests."""
    import time
    start_time = time.time()
    
    data = request.json
    code = data.get('code', '')
    mode = data.get('mode', 'source')
    target = data.get('target', 'hexagon')
    hvx_width = data.get('hvx_width', 128)
    
    try:
        ctx = TranslationContext(target_platform=target)
        
        if mode == 'source':
            transformer = CUDAToRIPPLETransformer(ctx)
            output = transformer.transform(code)
        else:
            translator = CUDAIRToRIPPLETranslator(ctx)
            output = translator.translate(code)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'output': output,
            'warnings': ctx.warnings,
            'time_ms': elapsed_ms
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze CUDA code without translation."""
    data = request.json
    code = data.get('code', '')
    
    try:
        lexer = CUDALexer(code)
        tokens = lexer.tokenize()
        
        analysis = {
            'lines': len(code.split('\n')),
            'tokens': len(tokens),
            'has_kernels': '__global__' in code,
            'uses_shared': '__shared__' in code,
            'uses_atomics': 'atomic' in code,
        }
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


# =============================================================================
# Main
# =============================================================================

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the web server."""
    print(f"🚀 Starting CUDA to RIPPLE Web Interface")
    print(f"   Open http://localhost:{port} in your browser")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.debug)
