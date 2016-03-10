#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"

void THNN_Byteunfolded_copy(THByteTensor *finput, THByteTensor *input,
                                  int kW, int kH,
                                  int dW, int dH,
                                  int padW, int padH,
                                  int nInputPlane,
                                  int inputWidth, int inputHeight,
                                  int outputWidth, int outputHeight)
{
  long k;
  uint8_t *input_data = THByteTensor_data(input);
  uint8_t *finput_data = THByteTensor_data(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    long nip = k / (kH*kW);
    long rest = k % (kH*kW);
    long kh = rest / kW;
    long kw = rest % kW;
    long x,y;
    long ix,iy;
    uint8_t *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    uint8_t *src = input_data + nip*(inputHeight*inputWidth);
    if (padW > 0 || padH > 0) {
      long lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = (y*dH - padH + kh);
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+y*outputWidth, 0, sizeof(uint8_t)*outputWidth);
        } else {
          if (dW==1){
             ix = (0 - padW + kw);
             lpad = fmaxf(0,padW-kw);
             rpad = fmaxf(0,padW-(kW-kw-1));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+(y*outputWidth), 0, sizeof(uint8_t)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+y*outputWidth, 0, sizeof(uint8_t)*lpad);
                memcpy(dst+(y*outputWidth+lpad), src+(iy*inputWidth+ix+lpad), sizeof(uint8_t)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, 0, sizeof(uint8_t)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = (x*dW - padW + kw);
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+(y*outputWidth+x), 0, sizeof(uint8_t)*1);
               else
                 memcpy(dst+(y*outputWidth+x), src+(iy*inputWidth+ix), sizeof(uint8_t)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = (y*dH + kh);
        ix = (0 + kw);
        if (dW == 1)
           memcpy(dst+(y*outputWidth), src+(iy*inputWidth+ix), sizeof(uint8_t)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+(y*outputWidth+x), src+(iy*inputWidth+ix+x*dW), sizeof(uint8_t)*(1));
         }
      }
    }
  }
}

static void THNN_ByteSpatialConvolutionMM_updateOutput_frame(THByteTensor *input, THByteTensor *output,
                                                            THByteTensor *weight, THByteTensor *bias, THByteTensor *finput,
                                                            int kW, int kH, int dW, int dH, int padW, int padH,
                                                            long nInputPlane, long inputWidth, long inputHeight,
                                                            long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THByteTensor *output2d;

  THNN_Byteunfolded_copy(finput, input, kW, kH, dW, dH, padW, padH,
                        nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d = THByteTensor_newWithStorage2d(output->storage, output->storageOffset,
                                           nOutputPlane, -1,
                                           outputHeight*outputWidth, -1);

  THByteTensor_mm8(output2d, weight, finput, 0, 0, 0, 1, 0);

  for(i = 0; i < nOutputPlane; i++)
    THByteVector_cadd8(output->storage->data+output->storageOffset+output->stride[0]*i,
                      THByteTensor_get1d(bias, i), outputHeight*outputWidth);

  THByteTensor_free(output2d);
}

void THNN_ByteSpatialConvolutionMM_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          THByteTensor *weight,
          THByteTensor *bias,
          THByteTensor *finput,
          THByteTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");


  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  if(input->nDimension == 3)
  {
    THByteTensor_resize2d(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THByteTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);

    THNN_ByteSpatialConvolutionMM_updateOutput_frame(input, output, weight, bias, finput,
                                                    kW, kH, dW, dH, padW, padH,
                                                    nInputPlane, inputWidth, inputHeight,
                                                    nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THByteTensor_resize3d(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THByteTensor_resize4d(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THByteTensor *input_t = THByteTensor_newSelect(input, 0, t);
      THByteTensor *output_t = THByteTensor_newSelect(output, 0, t);
      THByteTensor *finput_t = THByteTensor_newSelect(finput, 0, t);

      THNN_ByteSpatialConvolutionMM_updateOutput_frame(input_t, output_t, weight, bias, finput_t,
                                                      kW, kH, dW, dH, padW, padH,
                                                      nInputPlane, inputWidth, inputHeight,
                                                      nOutputPlane, outputWidth, outputHeight);

      THByteTensor_free(input_t);
      THByteTensor_free(output_t);
      THByteTensor_free(finput_t);
    }
  }
}

#endif
