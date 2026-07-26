import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:google_fonts/google_fonts.dart';

void main() {
  runApp(const CropYieldPredictorApp());
}

class CropYieldPredictorApp extends StatelessWidget {
  const CropYieldPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgriPredict',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          primary: const Color(0xFF2E7D32),
          secondary: const Color(0xFF81C784),
          surface: Colors.white,
          background: const Color(0xFFF4F7F6),
        ),
        textTheme: GoogleFonts.poppinsTextTheme(Theme.of(context).textTheme),
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
  
  // Controllers
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
    if (!_formKey.currentState!.validate()) return;
    
    setState(() => _isLoading = true);

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
        if (mounted) _showResultDialog(true, yieldVal.toStringAsFixed(2));
      } else {
        if (mounted) _showResultDialog(false, 'Server Error: ${response.statusCode}');
      }
    } catch (e) {
      if (mounted) _showResultDialog(false, 'Connection Error. Please try again.');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showResultDialog(bool success, String message) {
    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: success ? Colors.green.withOpacity(0.1) : Colors.red.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  success ? Icons.check_circle_outline : Icons.error_outline,
                  color: success ? Colors.green : Colors.red,
                  size: 48,
                ),
              ),
              const SizedBox(height: 24),
              Text(
                success ? 'Prediction Complete' : 'Oops!',
                style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              if (success) ...[
                const Text('Estimated Yield:', style: TextStyle(color: Colors.grey, fontSize: 16)),
                const SizedBox(height: 8),
                Text(
                  '$message Tons/Ha',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w800,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ] else ...[
                Text(
                  message,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16, color: Colors.grey),
                ),
              ],
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Done', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              )
            ],
          ),
        ),
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

  Widget _buildSectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16.0),
      child: Row(
        children: [
          Icon(icon, color: Theme.of(context).colorScheme.primary, size: 24),
          const SizedBox(width: 12),
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInputField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: TextFormField(
        controller: controller,
        keyboardType: keyboardType,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, color: Colors.grey.shade600),
          filled: true,
          fillColor: Colors.grey.shade50,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Colors.grey.shade300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide(color: Theme.of(context).colorScheme.primary, width: 2),
          ),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        ),
        validator: (val) => val == null || val.isEmpty ? 'This field is required' : null,
      ),
    );
  }

  Widget _buildSwitch({
    required String title,
    required bool value,
    required Function(bool) onChanged,
    required IconData icon,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: SwitchListTile(
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
        secondary: Icon(icon, color: Colors.grey.shade600),
        value: value,
        onChanged: onChanged,
        activeColor: Theme.of(context).colorScheme.primary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 160.0,
            floating: false,
            pinned: true,
            flexibleSpace: FlexibleSpaceBar(
              title: const Text(
                'AgriPredict',
                style: TextStyle(fontWeight: FontWeight.w700, color: Colors.white),
              ),
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Theme.of(context).colorScheme.primary,
                      Theme.of(context).colorScheme.secondary,
                    ],
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned(
                      right: -50,
                      top: -50,
                      child: Icon(Icons.eco, size: 200, color: Colors.white.withOpacity(0.15)),
                    ),
                  ],
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Form(
              key: _formKey,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Enter your farm details to get an accurate crop yield estimate.',
                      style: TextStyle(color: Colors.grey, fontSize: 16),
                    ),
                    const SizedBox(height: 24),

                    // Environmental Conditions
                    _buildSectionHeader('Environmental', Icons.wb_sunny_outlined),
                    _buildInputField(
                      controller: _rainfallCtrl,
                      label: 'Rainfall (mm)',
                      icon: Icons.water_drop_outlined,
                      keyboardType: const TextInputType.numberWithOptions(decimal: true),
                    ),
                    _buildInputField(
                      controller: _tempCtrl,
                      label: 'Temperature (°C)',
                      icon: Icons.thermostat_outlined,
                      keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
                    ),
                    _buildInputField(
                      controller: _weatherCtrl,
                      label: 'Weather Condition',
                      icon: Icons.cloud_outlined,
                    ),

                    // Farm Management
                    _buildSectionHeader('Farm Management', Icons.agriculture_outlined),
                    _buildSwitch(
                      title: 'Fertilizer Used',
                      value: _fertilizerUsed,
                      onChanged: (val) => setState(() => _fertilizerUsed = val),
                      icon: Icons.science_outlined,
                    ),
                    _buildSwitch(
                      title: 'Irrigation Used',
                      value: _irrigationUsed,
                      onChanged: (val) => setState(() => _irrigationUsed = val),
                      icon: Icons.water_outlined,
                    ),

                    // Crop & Soil
                    _buildSectionHeader('Crop Details', Icons.grass_outlined),
                    _buildInputField(
                      controller: _cropCtrl,
                      label: 'Crop Name',
                      icon: Icons.local_florist_outlined,
                    ),
                    _buildInputField(
                      controller: _soilCtrl,
                      label: 'Soil Type',
                      icon: Icons.landscape_outlined,
                    ),
                    _buildInputField(
                      controller: _regionCtrl,
                      label: 'Region',
                      icon: Icons.map_outlined,
                    ),
                    _buildInputField(
                      controller: _daysCtrl,
                      label: 'Days to Harvest',
                      icon: Icons.calendar_today_outlined,
                      keyboardType: const TextInputType.numberWithOptions(decimal: true),
                    ),

                    const SizedBox(height: 40),
                    
                    // Predict Button
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: FilledButton(
                        onPressed: _isLoading ? null : _submitForm,
                        style: FilledButton.styleFrom(
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                          elevation: 2,
                        ),
                        child: _isLoading
                            ? const SizedBox(
                                height: 24,
                                width: 24,
                                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5),
                              )
                            : const Text(
                                'Calculate Yield',
                                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                              ),
                      ),
                    ),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
