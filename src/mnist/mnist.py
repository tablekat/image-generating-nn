
with open("train-images.idx3-ubyte", "rb") as imgs:
  with open("train-labels.idx1-ubyte", "rb") as labels:
    with open("mnistout.csv", "w") as outf:

      for x in range(8):
        # drop headers
        imgs.read(1)
        labels.read(1)
      for x in range(8):
        imgs.read(1)
        
      num = 60000
      w = 28
      h = 28
      
      outf.write(str(num) + "," + str(w) + "," + str(h) + "\n")
      
      for i in range(num):
        if i % 25 == 0: print(i)
        label = ord(labels.read(1))
        outf.write(str(label) + ",")
        for p in range(w * h):
          pixel = ord(imgs.read(1))
          outf.write(str(pixel) + ",")
        outf.write("\n")