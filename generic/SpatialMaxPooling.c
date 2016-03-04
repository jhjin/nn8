static void THNN_ByteSpatialMaxPooling_updateOutput_frame(uint8_t *input_p, uint8_t *output_p,
                                                         uint8_t *ind_p,
                                                         long nslices,
                                                         long iwidth, long iheight,
                                                         long owidth, long oheight,
                                                         int kW, int kH, int dW, int dH,
                                                         int padW, int padH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    uint8_t *ip = input_p   + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * dH - padH;
        long wstart = j * dW - padW;
        long hend = fminf(hstart + kH, iheight);
        long wend = fminf(wstart + kW, iwidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        /* local pointers */
        uint8_t *op = output_p  + k*owidth*oheight + i*owidth + j;
        uint8_t *indp = ind_p   + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        uint8_t maxval = 0;
        long tcntr = 0;
        long x,y;
        for(y = hstart; y < hend; y++)
        {
          for(x = wstart; x < wend; x++)
          {
            tcntr = y*iwidth + x;
            uint8_t val = *(ip + tcntr);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max */
        *indp = maxindex + 1;
      }
    }
  }
}

void THNN_ByteSpatialMaxPooling_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          THByteTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  uint8_t *input_data;
  uint8_t *output_data;
  uint8_t *indices_data;


  THArgCheck(input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2, "input image smaller than kernel size");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  if (ceil_mode)
  {
    oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }
  else
  {
    oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  /* get contiguous input */
  input = THByteTensor_newContiguous(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THByteTensor_resize3d(output, nslices, oheight, owidth);
    /* indices will contain the locations for each output point */
    THByteTensor_resize3d(indices,  nslices, oheight, owidth);

    input_data = THByteTensor_data(input);
    output_data = THByteTensor_data(output);
    indices_data = THByteTensor_data(indices);

    THNN_ByteSpatialMaxPooling_updateOutput_frame(input_data, output_data,
                                                 indices_data,
                                                 nslices,
                                                 iwidth, iheight,
                                                 owidth, oheight,
                                                 kW, kH, dW, dH,
                                                 padW, padH);
  }
  else
  {
    long p;

    THByteTensor_resize4d(output, nbatch, nslices, oheight, owidth);
    /* indices will contain the locations for each output point */
    THByteTensor_resize4d(indices, nbatch, nslices, oheight, owidth);

    input_data = THByteTensor_data(input);
    output_data = THByteTensor_data(output);
    indices_data = THByteTensor_data(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_ByteSpatialMaxPooling_updateOutput_frame(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
                                                   indices_data+p*nslices*owidth*oheight,
                                                   nslices,
                                                   iwidth, iheight,
                                                   owidth, oheight,
                                                   kW, kH, dW, dH,
                                                   padW, padH);
    }
  }

  /* cleanup */
  THByteTensor_free(input);
}
