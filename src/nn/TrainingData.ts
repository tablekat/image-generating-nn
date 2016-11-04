
import { Util } from './Util'

export class TrainingPair{
  public input: Float64Array;
  public output: Float64Array;
  constructor(input: Float64Array | number[], output: Float64Array | number[]){
    this.input = new Float64Array(input);
    this.output = new Float64Array(output);
  }
}

export class TrainingData{

  public data: TrainingPair[];
  public batchIterator: number;

  constructor(){
    this.data = [];
    this.batchIterator = 0;
  }

  public export(){
    return JSON.stringify(this.data.map(x => {
      return { input: x.input, output: x.output }
    }));
  }
  public import(data: string){
    var d = JSON.parse(data);
    d.map(x => this.train(x.input, x.output));
  }

  public train(input: number[], output: number[]){
    this.data.push(new TrainingPair(input, output));
  }

  public shuffle(){
    this.data = Util.shuffle(this.data);
  }

  public reset(){
    this.batchIterator = 0;
    this.shuffle();
  }

  public getBatch(batchSize: number): TrainingPair[]{
    var result = [];
    for(var i=0; i < batchSize; ++i){
      if(i + this.batchIterator > this.data.length - 1) break;
      result.push(this.data[i + this.batchIterator]);
    }
    this.batchIterator += batchSize;
    return result;
  }

  public split(ratio: number){
    if(ratio < 0 || ratio > 1) return new TrainingData();
    var removeCount = Math.floor(this.data.length * ratio);
    var keepCount = this.data.length - removeCount;
    var newData = new TrainingData();
    for(var i=0; i < removeCount; ++i){
      newData.data.push(this.data[keepCount + i]);
    }

    this.data.length = keepCount;
    return newData;
  }
}
