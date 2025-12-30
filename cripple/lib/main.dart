import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const CrippleApp());
}

class CrippleApp extends StatelessWidget {
  const CrippleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cripple',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        textTheme: GoogleFonts.interTextTheme(),
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C63FF),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF121212),
        useMaterial3: true,
        textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController _inputController = TextEditingController();
  String _outputCode = "";
  bool _isTranslating = false;
  String _statusMessage = "";

  Future<void> _translateCode() async {
    final sourceCode = _inputController.text;
    if (sourceCode.trim().isEmpty) {
      setState(() {
        _statusMessage = "Please enter some CUDA code first.";
      });
      return;
    }

    setState(() {
      _isTranslating = true;
      _outputCode = "";
      _statusMessage = "Translating...";
    });

    try {
      // Accessing localhost from the browser/emulator
      // For web, localhost points to the same machine.
      // For Android emulator, use 10.0.2.2. For iOS simulator, localhost is fine.
      final response = await http.post(
        Uri.parse('http://127.0.0.1:5001/translate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'source': sourceCode}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _outputCode = data['translated'] ?? "// No output returned";
          _statusMessage = "Translation successful!";
        });
      } else {
        setState(() {
          _outputCode = "// Error: ${response.statusCode} - ${response.reasonPhrase}";
           _statusMessage = "Translation failed.";
        });
        print("Error: ${response.body}");
      }
    } catch (e) {
      setState(() {
        _outputCode = "// Connection Error: $e\n// Make sure the Python server is running.";
        _statusMessage = "Connection error.";
      });
    } finally {
      setState(() {
        _isTranslating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Cripple',
          style: GoogleFonts.jetBrainsMono(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    "CUDA Input",
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          color: Theme.of(context).colorScheme.primary,
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                ),
                if (_statusMessage.isNotEmpty)
                   Text(
                    _statusMessage,
                     style: TextStyle(
                       color: _statusMessage.contains("Error") || _statusMessage.contains("failed") ? Colors.redAccent : Colors.greenAccent,
                       fontSize: 12
                     ),
                  )
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: Theme.of(context).colorScheme.outlineVariant.withValues(alpha: 0.2),
                  ),
                ),
                child: TextField(
                  controller: _inputController,
                  maxLines: null,
                  expands: true,
                  style: GoogleFonts.jetBrainsMono(fontSize: 14),
                  decoration: const InputDecoration(
                    hintText: "__global__ void cudaKernel(...) { ... }",
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.all(16),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Center(
              child: FilledButton.icon(
                onPressed: _isTranslating ? null : _translateCode,
                icon: _isTranslating
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.transform),
                label: Text(_isTranslating ? "Translating..." : "Translate to Ripple"),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "Ripple Output",
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: Theme.of(context).colorScheme.secondary,
                        fontWeight: FontWeight.bold,
                      ),
                ),
                if (_outputCode.isNotEmpty)
                  IconButton(
                    icon: const Icon(Icons.copy_all, size: 20),
                    tooltip: 'Copy to clipboard',
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: _outputCode));
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Ripple code copied to clipboard!'),
                          duration: Duration(seconds: 2),
                          backgroundColor: Color(0xFF6C63FF),
                        ),
                      );
                    },
                  ),
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: Theme.of(context).colorScheme.outlineVariant.withValues(alpha: 0.2),
                  ),
                ),
                padding: const EdgeInsets.all(16),
                child: SingleChildScrollView(
                  child: SelectableText(
                    _outputCode.isEmpty ? "// Output will appear here..." : _outputCode,
                    style: GoogleFonts.jetBrainsMono(
                      fontSize: 14,
                      color: _outputCode.isEmpty ? Colors.grey : const Color(0xFFA5D6A7),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
