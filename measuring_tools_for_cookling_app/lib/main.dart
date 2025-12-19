
// Clean, single-definition main.dart
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter_plus/tflite_flutter_plus.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Measuring Tools for Cooking Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.deepOrange,
        scaffoldBackgroundColor: Colors.transparent, // Important for gradient background
        fontFamily: 'Roboto', // Default, but explicit is good
        textTheme: const TextTheme(
          displayLarge: TextStyle(fontWeight: FontWeight.bold, fontSize: 32, letterSpacing: -1.0),
          titleLarge: TextStyle(fontWeight: FontWeight.bold, fontSize: 22),
          bodyLarge: TextStyle(fontSize: 16),
        ),
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  List? _output;
  String? _predictedLabel;
  double? _confidence;
  final picker = ImagePicker();
  Interpreter? _interpreter;
  List<String>? _labels;
  bool _modelLoaded = false;
  bool _labelsLoaded = false;
  final Map<String, int> _analyticsData = {};
  int _totalScans = 0;

  @override
  void initState() {
    super.initState();
    loadModel();
    WidgetsBinding.instance.addPostFrameCallback((_) => loadLabels());
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  Future<void> loadModel() async {
    try {
      final interpreter = await Interpreter.fromAsset('converted_tflite.tflite');
      setState(() {
        _interpreter = interpreter;
        _modelLoaded = true;
      });
      try {
        final inputTensor = _interpreter!.getInputTensor(0);
        final outputTensor = _interpreter!.getOutputTensor(0);
        print('Model loaded: input shape=${inputTensor.shape} type=${inputTensor.type}, output shape=${outputTensor.shape} type=${outputTensor.type}');
      } catch (e) {
        print('Model loaded but failed to query tensors: $e');
      }
    } catch (e, st) {
      print('Failed to load model: $e\n$st');
      setState(() => _modelLoaded = false);
    }
  }

  Future<void> loadLabels() async {
    try {
      final labelsData = await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
      final labels = labelsData.split('\n').where((l) => l.isNotEmpty).toList();
      setState(() {
        _labels = labels;
        _labelsLoaded = true;
      });
      print('Labels loaded: ${labels.length} entries');
    } catch (e, st) {
      print('Failed to load labels: $e\n$st');
      setState(() => _labelsLoaded = false);
    }
  }

  Future<void> pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() => _image = File(pickedFile.path));
      runModelOnImage(_image!);
    }
  }

  Future<void> captureImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() => _image = File(pickedFile.path));
      runModelOnImage(_image!);
    }
  }

  void runModelOnImage(File image) {
    if (_interpreter == null || _labels == null) {
      print('Interpreter or labels not ready. _modelLoaded=$_modelLoaded _labelsLoaded=$_labelsLoaded');
      return;
    }

    try {
      final imageBytes = image.readAsBytesSync();
      final imageDecoded = img.decodeImage(imageBytes);
      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape; 
      final inputType = inputTensor.type;

      final height = inputShape.length >= 3 ? inputShape[inputShape.length - 3] : 224;
      final width = inputShape.length >= 2 ? inputShape[inputShape.length - 2] : 224;

      final resizedImage = img.copyResizeCropSquare(imageDecoded!, size: width);

      dynamic input;
      if (inputType == TfLiteType.uint8 || inputType == TfLiteType.int8) {
        input = List.generate(1, (_) => List.generate(width, (y) => List.generate(width, (x) {
          final p = resizedImage.getPixel(x, y);
          return [p.r, p.g, p.b];
        })));
      } else {
        input = List.generate(1, (_) => List.generate(width, (y) => List.generate(width, (x) {
          final p = resizedImage.getPixel(x, y);
          return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
        })));
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outShape = outputTensor.shape; // e.g. [1,N]
      final outSize = outShape.reduce((a, b) => a * b);
      // ignore: prefer_const_constructors
      final output = [List<double>.filled(outSize, 0.0)];

      print('Running inference: inputType=$inputType inputShape=$inputShape outShape=$outShape outSize=$outSize');

      _interpreter!.run(input, output);

      final scores = (output[0] as List).map((e) => (e is int) ? e.toDouble() : (e as num).toDouble()).toList();
      var maxIdx = 0;
      for (var i = 1; i < scores.length; i++) {
        if (scores[i] > scores[maxIdx]) maxIdx = i;
      }
      final label = (_labels != null && maxIdx < _labels!.length) ? _labels![maxIdx] : 'Unknown';
      final confidence = scores[maxIdx];

      print('Inference done. top=$label confidence=$confidence');

      setState(() {
        _output = output;
        _predictedLabel = label;
        _confidence = confidence;
        _totalScans++;
        if (!_analyticsData.containsKey(label)) {
          _analyticsData[label] = 0;
        }
        _analyticsData[label] = _analyticsData[label]! + 1;
      });
    } catch (e, st) {
      print('Interpreter run failed: $e\n$st');
    }
  }

  void _showAnalytics() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        maxChildSize: 0.9,
        minChildSize: 0.4,
        builder: (_, controller) => Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(2)),
                ),
              ),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                   const Text('Scan Analytics', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.deepOrange)),
                   Container(
                     padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                     decoration: BoxDecoration(color: Colors.deepOrange.withOpacity(0.1), borderRadius: BorderRadius.circular(20)),
                     child: Text('Total: $_totalScans', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.deepOrange[800])),
                   )
                ],
              ),
              const SizedBox(height: 24),
              Expanded(
                child: _analyticsData.isEmpty
                    ? Center(child: Text("No scans yet. Start identifying tools!", style: TextStyle(color: Colors.grey[400])))
                    : ListView.builder(
                        controller: controller,
                        itemCount: _analyticsData.length,
                        itemBuilder: (context, index) {
                          final entry = _analyticsData.entries.toList()[index];
                          final percentage = (entry.value / _totalScans);
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(entry.key, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                                    Text('${(percentage * 100).toStringAsFixed(0)}%', style: TextStyle(color: Colors.grey[600])),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Stack(
                                  children: [
                                    Container(height: 10, decoration: BoxDecoration(color: Colors.grey[100], borderRadius: BorderRadius.circular(5))),
                                    FractionallySizedBox(
                                      widthFactor: percentage,
                                      child: Container(height: 10, decoration: BoxDecoration(gradient: const LinearGradient(colors: [Colors.orange, Colors.deepOrange]), borderRadius: BorderRadius.circular(5))),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Align(alignment: Alignment.centerRight, child: Text('${entry.value} scans', style: TextStyle(fontSize: 12, color: Colors.grey[500]))),
                              ],
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Colors.orange.shade50,
            Colors.amber.shade50,
            Colors.white,
          ],
        ),
      ),
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Measuring Tools', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.deepOrange)),
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
          actions: [
            Container(
              margin: const EdgeInsets.only(right: 16),
              decoration: BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(color: Colors.orange.withOpacity(0.2), blurRadius: 8, offset: const Offset(0, 4))],
              ),
              child: IconButton(
                icon: const Icon(Icons.bar_chart_rounded, color: Colors.deepOrange),
                onPressed: _showAnalytics,
                tooltip: 'View Analytics',
              ),
            )
          ],
        ),
        body: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Card(
                  elevation: 12,
                  shadowColor: Colors.deepOrange.withOpacity(0.3),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(32)),
                  clipBehavior: Clip.antiAlias,
                  child: Container(
                    width: 320,
                    height: 320,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      border: Border.all(color: Colors.white, width: 4),
                    ),
                    child: _image == null
                        ? Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                padding: const EdgeInsets.all(24),
                                decoration: BoxDecoration(
                                  color: Colors.orange.shade50,
                                  shape: BoxShape.circle,
                                ),
                                child: Icon(Icons.blender_rounded, size: 64, color: Colors.deepOrange.shade300),
                              ),
                              const SizedBox(height: 24),
                              Text('Ready to classify', style: TextStyle(color: Colors.grey[400], fontSize: 18, fontWeight: FontWeight.w500)),
                            ],
                          )
                        : Image.file(_image!, fit: BoxFit.cover),
                  ),
                ),
                const SizedBox(height: 32),
                if (_predictedLabel == null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 48.0),
                    child: Text(
                      'Take a photo of a measuring cup, scale, or spoon!',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 18, color: Colors.grey[600], height: 1.5),
                    ),
                  )
                else
                  Container(
                    margin: const EdgeInsets.symmetric(horizontal: 24),
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [BoxShadow(color: Colors.orange.withOpacity(0.1), blurRadius: 16, offset: const Offset(0, 8))],
                    ),
                    child: Column(
                      children: [
                        const Text('IDENTIFIED AS', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey, letterSpacing: 1.5)),
                        const SizedBox(height: 8),
                        Text(
                           _predictedLabel!,
                          style: const TextStyle(
                            fontSize: 32,
                            color: Colors.deepOrange,
                            fontWeight: FontWeight.w900,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        if (_confidence != null) ...[
                          const SizedBox(height: 16),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: LinearProgressIndicator(
                              value: _confidence,
                              backgroundColor: Colors.grey[100],
                              color: _confidence! > 0.8 ? Colors.green : Colors.orange,
                              minHeight: 12,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                            style: TextStyle(color: Colors.grey[500], fontSize: 13, fontWeight: FontWeight.w600),
                          ),
                        ]
                      ],
                    ),
                  ),
                const SizedBox(height: 48),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildActionButton(
                      onPressed: pickImage,
                      icon: Icons.photo_library_rounded,
                      label: 'Gallery',
                      isPrimary: false,
                    ),
                    const SizedBox(width: 24),
                    _buildActionButton(
                      onPressed: captureImage,
                      icon: Icons.camera_alt_rounded,
                      label: 'Scan',
                      isPrimary: true,
                    ),
                  ],
                ),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildActionButton({required VoidCallback onPressed, required IconData icon, required String label, required bool isPrimary}) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(24),
      child: Container(
        width: 130, // Fixed width for symmetry
        padding: const EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          gradient: isPrimary ? const LinearGradient(colors: [Colors.orange, Colors.deepOrange], begin: Alignment.topLeft, end: Alignment.bottomRight) : null,
          color: isPrimary ? null : Colors.white,
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
             BoxShadow(
               color: isPrimary ? Colors.deepOrange.withOpacity(0.4) : Colors.grey.withOpacity(0.1),
               blurRadius: 12,
               offset: const Offset(0, 6)
             )
          ],
        ),
        child: Column(
          children: [
            Icon(icon, size: 32, color: isPrimary ? Colors.white : Colors.deepOrange),
            const SizedBox(height: 8),
            Text(label, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: isPrimary ? Colors.white : Colors.grey[800])),
          ],
        ),
      ),
    );
  }
}
