var path = require('path');
var webpack = require('webpack');

module.exports = {
  entry: [
    './src/image/main'
  ],
  devtool: 'eval-source-map',
  output: {
    path: __dirname + '/build/',
    filename: 'app.js',
    //publicPath: './build/'
  },
  module: {
    loaders: [
      {
        test: /\.ts$/,
        loader: 'ts',
        exclude:/(node_modules|releases)/
      },
    ]
  },

  resolve: {
    extensions: ['.ts','.js','.jsx','.json', '.css', '.html']
  },
};
