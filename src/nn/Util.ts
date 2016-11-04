

export class Util {
  public static clamp(arg, min, max){
    if(arg < min) return min;
    if(arg > max) return max;
    return arg;
  }

  public static sigmoid(z){
    return 1 / (1 + Math.exp(-z));
  }

  public static sigmoidPrime(z){
    return Util.sigmoid(z) * (1 - Util.sigmoid(z));
  }

  public static dot(xs, ys){
    if(xs.length != ys.length){
      throw new Error("Util.dot: xs and ys not same length.");
    }
    var z = 0;
    for(var i=0; i < xs.length; ++i){
      z += xs[i] * ys[i];
    }
    return z;
  }

  public static hadamard(xs, ys){
    if(xs.length != ys.length) throw new Error("Util.hadamard: xs and ys not same length");
    var res = new Float64Array(xs.length);
    for(var i=0; i < xs.length; ++i){
      res[i] = xs[i] * ys[i];
    }
    return res;
  }

  public static rand(){
    return Math.random() - Math.random();
  }

  public static shuffle(o){
    for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
    return o;
  }
}
