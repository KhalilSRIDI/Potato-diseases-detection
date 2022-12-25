import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class Tensorflow extends StatefulWidget {
  @override
  _TensorflowState createState() => _TensorflowState();
}

class _TensorflowState extends State<Tensorflow> {
  List _recognitions;
  List _binary;
  File _image;
  bool _busy = false;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _loading = true;

    loadModel().then((value) {
      setState(() {
        _loading = false;
      });
    });
  }

  loadModel() async {
    await Tflite.loadModel(
      model: "assets/model.tflite",
      labels: "assets/labels.txt",
      numThreads: 1,
    );
  }

  classifyImage(File image) async {
    img.Image image2 = img.decodeImage(image.readAsBytesSync());

    var recognitions = await Tflite.runModelOnImage(
        path: image.path,
        imageMean: 0,
        imageStd: 1,
        numResults: 3,
        threshold: 0.05,
        asynch: true);
    setState(() {
      _loading = false;
      _recognitions = recognitions;
    });
    print(_recognitions);
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  Uint8List imageToByteListFloat32(
      img.Image imge, int inputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = imge.getPixel(j, i);
        buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  pickImage() async {
    ImagePicker _picker = ImagePicker();
    var image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return null;
    setState(() {
      _busy = true;
      _loading = true;
      _image = File(image.path);
    });
    classifyImage(_image);
  }

  pickImageCam() async {
    ImagePicker _picker = ImagePicker();
    var image = await _picker.pickImage(source: ImageSource.camera);
    if (image == null) return null;
    setState(() {
      _busy = true;
      _loading = true;
      _image = File(image.path);
    });
    classifyImage(_image);
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: Text(
            "Potato Disease Detection",
            style: TextStyle(color: Colors.white, fontSize: 25),
          ),
          backgroundColor: Color.fromARGB(255, 19, 142, 190),
          elevation: 0,
        ),
        body: Container(
          // height: MediaQuery.of(context).size.height,
          color: Colors.black45,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _loading
                  ? Container(
                      height: 300,
                      width: 300,
                    )
                  : Container(
                      margin: EdgeInsets.all(20),
                      width: MediaQuery.of(context).size.width,
                      //height: MediaQuery.of(context).size.height,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          _image == null ? Container() : Image.file(_image),
                          SizedBox(
                            height: 20,
                          ),
                          _image == null
                              ? Container()
                              : _recognitions != null
                                  ? Text(
                                      _recognitions[0]["label"],
                                      style: TextStyle(
                                          color: Colors.black, fontSize: 20),
                                    )
                                  : Container(child: Text(""))
                        ],
                      ),
                    ),
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.01,
              ),
              GestureDetector(
                onTap: pickImageCam, //no parenthesis
                child: Container(
                  width: MediaQuery.of(context).size.width - 200,
                  alignment: Alignment.center,
                  padding: EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                  decoration: BoxDecoration(
                      color: Colors.blueGrey[600],
                      borderRadius: BorderRadius.circular(15)),
                  child: Text(
                    'Take A Photo',
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
              ),
              SizedBox(
                height: 30,
              ),
              GestureDetector(
                onTap: pickImage, //no parenthesis
                child: Container(
                  width: MediaQuery.of(context).size.width - 200,
                  alignment: Alignment.center,
                  padding: EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                  decoration: BoxDecoration(
                      color: Colors.blueGrey[600],
                      borderRadius: BorderRadius.circular(15)),
                  child: Text(
                    'Pick From Gallery',
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
