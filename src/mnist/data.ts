
import * as path from 'path';
import * as fs from 'fs';
import * as readline from 'readline';
import * as Promise from 'bluebird';

export class NumberImg {
  public realValue: number;
  public imgData: Uint8Array;
  constructor(line: string){
    if(line[line.length - 1] == ',') line = line.substring(0, line.length - 1);
    var vals: any[] = line.split(',');
    this.realValue = parseInt(vals[1]);
    this.imgData = new Uint8Array(vals.slice(1));
  }
  public valToArr(): Array<number>{
    return (<any>new Array(10)).fill(0).map((x, i) => i == this.realValue ? 1 : 0);
  }
}

export class Data {
  public totalCount: number;
  public width: number;
  public height: number;
  public imgs: NumberImg[] = [];

  constructor(){}

  public static loadFile(fpath: string): Promise<Data>{
    var instream = fs.createReadStream(fpath);
    return new Promise<Data>((resolve) => {
      var self = new Data();
      var lineReader = readline.createInterface({ input: instream });
      var firstLine = true;

      lineReader.on('line', (line) => {
        if(firstLine){
          [self.totalCount, self.width, self.height] = line.split(',');
          firstLine = false;
        }else{
          if(!line.trim()) return;
          self.imgs.push(new NumberImg(line));
        }
      });

      lineReader.on('close', () => resolve(self));
    })
  }
}
