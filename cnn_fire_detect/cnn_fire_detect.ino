#include "model.h" //Name of the Model !!!!!!!!!!!!!!!!!
#include "img.h" //image to test our model

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>




const int no_pixels = 6912;


// TensorFlow Lite for Microcontroller global variables
tflite::ErrorReporter *error_reporter = nullptr;//!!!!!!!!!!

const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
tflite::MicroErrorReporter tflu_error;

//100 for normal cnn 
constexpr int tensor_arena_size = 100 * 1024;//hyperparameter!!!!!
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));


/////////////////////////////////////////////////////////////////////
void tflu_initialization()// Model Initialization
{
  Serial.println("TFLu initialization - start");

  // Load the TFLITE model
  tflu_model = tflite::GetModel(TFLite_Models_model_tflite);//NAME OF THE MODEL BUT INSTEAD OF. USE _!!!!, path is also included in the name
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }

   tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);


   Serial.println("TFLu initialization - completed");
}




//////////////////////////////////////////////////////////////////////////

void setup() {
  Serial.begin(9600);
  while (!Serial);// wait for serial initialization


  tflu_initialization();
  delay(4000);
  Serial.println("Init is done");
}


////////////////////////////////////////////////////////

void loop() {
  unsigned long timeBegin = millis();
  
  
  // Initialize the input tensor
  for (int i = 0; i < no_pixels; i++) {
    tflu_i_tensor->data.f[i] = inp_t[i]/255;
  }


  // Run inference
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }
  

  float out_f = tflu_o_tensor->data.f[0];


   Serial.println(out_f);
   if(out_f > 0.5) {
     Serial.println("Not Fire");
   }
   else {
     Serial.println("Fire");
   }

  
  //execution time calculation
  unsigned long timeEnd = millis();
  unsigned long duration = timeEnd - timeBegin;
  double averageDuration = (double)duration / 1000.0;
  Serial.println(averageDuration);
  
  Serial.println();
  delay(4000);
}
