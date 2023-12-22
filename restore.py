
from dotenv import load_dotenv
load_dotenv()
import replicate

model =  replicate.models.get("tencentarc/gfpgan")
version =model.versions.get("9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3")

def predict_image(filename):

  output = replicate.run(
    "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
    input={
      "img": open(filename, "rb")
      
      }
      
    
  )
  
  output = version.predict(input)  
  print(output)
  return output