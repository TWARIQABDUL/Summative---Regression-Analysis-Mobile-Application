import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const CropYieldPredictorApp());
}

class CropYieldPredictorApp extends StatelessWidget {
  const CropYieldPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Yield Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  
  // Form controllers initialized with example data
  final TextEditingController _rainfallCtrl = TextEditingController(text: '750');
  final TextEditingController _tempCtrl = TextEditingController(text: '25.5');
  final TextEditingController _daysCtrl = TextEditingController(text: '110');
  final TextEditingController _regionCtrl = TextEditingController(text: 'North');
  final TextEditingController _soilCtrl = TextEditingController(text: 'Loam');
  final TextEditingController _cropCtrl = TextEditingController(text: 'Wheat');
  final TextEditingController _weatherCtrl = TextEditingController(text: 'Sunny');

  bool _fertilizerUsed = true;
  bool _irrigationUsed = true;
  
  bool _isLoading = false;

  Future<void> _submitForm() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    
    setState(() {
      _isLoading = true;
    });

    try {
      final Map<String, dynamic> payload = {
        "Rainfall_mm": double.tryParse(_rainfallCtrl.text) ?? 0.0,
        "Temperature_Celsius": double.tryParse(_tempCtrl.text) ?? 0.0,
        "Fertilizer_Used": _fertilizerUsed ? 1 : 0,
        "Irrigation_Used": _irrigationUsed ? 1 : 0,
        "Days_to_Harvest": double.tryParse(_daysCtrl.text) ?? 0.0,
        "Region": _regionCtrl.text,
        "Soil_Type": _soilCtrl.text,
        "Crop": _cropCtrl.text,
        "Weather_Condition": _weatherCtrl.text,
      };

      final response = await http.post(
        Uri.parse('https://summative-regression-analysis-mobile.onrender.com/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final yieldVal = data['predicted_yield_tons_per_hectare'];
        _showResultDialog('Prediction Success', 'Predicted Yield: $yieldVal tons/hectare');
      } else {
        _showResultDialog('Prediction Error', 'Server returned status: ${response.statusCode}\n${response.body}');
      }
    } catch (e) {
      _showResultDialog('Error', 'An error occurred: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showResultDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _rainfallCtrl.dispose();
    _tempCtrl.dispose();
    _daysCtrl.dispose();
    _regionCtrl.dispose();
    _soilCtrl.dispose();
    _cropCtrl.dispose();
    _weatherCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Predict Crop Yield'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: ListView(
            children: [
              TextFormField(
                controller: _rainfallCtrl,
                decoration: const InputDecoration(labelText: 'Rainfall (mm)'),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: _tempCtrl,
                decoration: const InputDecoration(labelText: 'Temperature (°C)'),
                keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              SwitchListTile(
                title: const Text('Fertilizer Used'),
                value: _fertilizerUsed,
                onChanged: (val) => setState(() => _fertilizerUsed = val),
              ),
              SwitchListTile(
                title: const Text('Irrigation Used'),
                value: _irrigationUsed,
                onChanged: (val) => setState(() => _irrigationUsed = val),
              ),
              TextFormField(
                controller: _daysCtrl,
                decoration: const InputDecoration(labelText: 'Days to Harvest'),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: _regionCtrl,
                decoration: const InputDecoration(labelText: 'Region'),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: _soilCtrl,
                decoration: const InputDecoration(labelText: 'Soil Type'),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: _cropCtrl,
                decoration: const InputDecoration(labelText: 'Crop'),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: _weatherCtrl,
                decoration: const InputDecoration(labelText: 'Weather Condition'),
                validator: (val) => val == null || val.isEmpty ? 'Required' : null,
              ),
              const SizedBox(height: 24),
              _isLoading
                  ? const Center(child: CircularProgressIndicator())
                  : ElevatedButton(
                      onPressed: _submitForm,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: const Text('Predict Yield', style: TextStyle(fontSize: 18)),
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
