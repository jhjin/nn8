static TH_INLINE void THByteVector_cadd8(uint8_t *y, const uint8_t c, const long n)
{
  long i = 0;

  for(;i < n-16; i += 16)
  {
    y[i] += c;
    y[i+1] += c;
    y[i+2] += c;
    y[i+3] += c;
    y[i+4] += c;
    y[i+5] += c;
    y[i+6] += c;
    y[i+7] += c;
    y[i+8] += c;
    y[i+9] += c;
    y[i+10] += c;
    y[i+11] += c;
    y[i+12] += c;
    y[i+13] += c;
    y[i+14] += c;
    y[i+15] += c;
  }

  for(; i < n; i++)
    y[i] += c;
}
