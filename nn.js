/*var NN = function(){
	this.input = [];
	this.firstWeights = [];
	this.hiddenLayer = [];
	this.secondWeights = [];
	this.output = [];
	
	this.fitness = 0;
	
	this.init = function(first_weights, second_weights){
		this.firstWeights = first_weights.slice(0);
		this.secondWeights = second_weights.slice(0);
	}
	
	this.sendInput = function(data){
		this.input = data.slice(0);
		this.input.push([-1]); //bias is important
	}	
	
	this.forwardProp = function(){
		this.hiddenLayer = matrixMult(this.input, this.firstWeights);
		this.hiddenLayer = arraySigmoid(this.hiddenLayer);
		this.hiddenLayer.push([-1]); //more bias
		this.output = matrixMult(this.hiddenLayer, this.secondWeights);
		this.output = arraySigmoid(this.output);
	}
};*/

var NN = function(){
//	this.LAYERS_SIZE = [784, 1500, 1500, 10];
	this.LAYERS_SIZE = [1, 7, 7, 7, 1];
	this.ALPHA = 0.01;
	
	this.nLayers = [];
	this.nWeights = [];
	
	this.nLayersRaw = [];
	this.nBackProp = [];
	this.nGradients = [];
	this.nTotalGradients = [];
	
	this.sendInput = function(data){
		this.nLayers[0] = data.slice(0); //must be size of this.LAYERS_SIZE[0]
		this.nLayers[0].push([-1]);
		
		this.nLayersRaw[0] = data.slice(0);
		this.nLayersRaw[0].push([-1]);
	}
	
	this.allocSpace = function(){ //run once before training
		//weights:
		for(var i = 1; i < this.LAYERS_SIZE.length; i++){
			var currWeights = [];
			for(var j = 0; j < this.LAYERS_SIZE[i]; j++){
				var column = [];
				for(var k = 0; k < this.LAYERS_SIZE[i-1]+1; k++){
					column.push(Math.random()*2-1);
				}
				currWeights.push(column.slice(0));
			}
			this.nWeights.push(currWeights);
		}
		
		for(var i = 0; i < this.LAYERS_SIZE.length; i++){
			this.nBackProp.push([]);
		}
	}
	
	this.resetGradients = function(){
		for(var i = 0; i < this.LAYERS_SIZE.length-1; i++){
			var matrixC = [];
			for(var j = 0; j < this.LAYERS_SIZE[i+1]; j++){
				var column = [];
				for(var k = 0; k < this.LAYERS_SIZE[i]+1; k++){
					column.push(0);
				}
				matrixC.push(column);
			}
			this.nTotalGradients[i] = matrixC;
		}
	}
	
	this.forwardProp = function(){
		for(var i = 1; i < this.LAYERS_SIZE.length; i++){
			this.nLayers[i] = matrixMult(this.nLayers[i-1], this.nWeights[i-1]);
			
			this.nLayersRaw[i] = matrixMult(this.nLayers[i-1], this.nWeights[i-1]);
			
			this.nLayers[i] = arraySigmoid(this.nLayers[i]);
			if(i != this.LAYERS_SIZE.length-1){ //don't add bias to output layer
				this.nLayers[i].push([-1]);
				this.nLayersRaw[i].push([-1]);
			}
		}
	}
	
	this.backwardProp = function(y){ //per one data point
		var yHat = this.nLayers[this.LAYERS_SIZE.length-1];
		var error = matrixSubtract(yHat, y);
		this.nBackProp[this.LAYERS_SIZE.length-1] = hProduct(error, arraySigmoidPrime(this.nLayersRaw[this.LAYERS_SIZE.length-1]));
		for(var i = this.LAYERS_SIZE.length-2; i >= 0; i--){	
			if(i == this.LAYERS_SIZE.length-2){
				this.nBackProp[i] = hProduct(matrixMult(this.nBackProp[i+1], flipMatrix(this.nWeights[i])), arraySigmoidPrime(this.nLayersRaw[i]));
			}else{
				this.nBackProp[i] = hProduct(matrixMult(this.nBackProp[i+1].slice(0, this.nBackProp[i+1].length-1), flipMatrix(this.nWeights[i])), arraySigmoidPrime(this.nLayersRaw[i]));
			}
		}
		
		for(var i = 0; i < this.LAYERS_SIZE.length-2; i++){
			this.nGradients[i] = matrixMult(flipMatrix(this.nLayers[i]), this.nBackProp[i+1].slice(0, this.nBackProp[i+1].length-1));
		}
		this.nGradients[this.LAYERS_SIZE.length-2] = matrixMult(flipMatrix(this.nLayers[this.LAYERS_SIZE.length-2]), this.nBackProp[this.LAYERS_SIZE.length-1]);
		
		for(var i = 0; i < this.LAYERS_SIZE.length-1; i++){
			this.nTotalGradients[i] = matrixAdd(this.nTotalGradients[i], this.nGradients[i]);
		}
	}
	
	this.adjust = function(){
		for(var i = 0; i < this.LAYERS_SIZE.length-1; i++){
			this.nWeights[i] = matrixSubtract(this.nWeights[i], scalarMult(this.nTotalGradients[i], this.ALPHA));
		}
	}
}

function matrixMult(matrixA, matrixB){
	if(matrixA.length == matrixB[0].length){
		var matrixC = [];
		//result.length = matrixB.length;
		for(var j = 0; j < matrixB.length; j++){
			var column = [];
			for(var i = 0; i < matrixA[0].length; i++){
				column.push(0);
			}
			matrixC.push(column.slice(0));
		}
		
		for(var x = 0; x < matrixC.length; x++){
			for(var y = 0; y < matrixC[0].length; y++){
				var total = 0;
				for(var i = 0; i < matrixA.length; i++){
					total += matrixA[i][y] * matrixB[x][i];
				} 
				matrixC[x][y] = total;
			}
		}
		
		return matrixC;
	}else{
		console.log("DAMN.");	
		return "Matrix sizes do not match.";
	}
	
}

function scalarMult(matrixA, b){
	var matrixC = [];
	
	for(var i = 0; i < matrixA.length; i++){
		var column = [];
		for(var j = 0; j < matrixA[0].length; j++){
			column.push(matrixA[i][j] * b);
		}
		matrixC.push(column.slice(0));
	}
	
	return matrixC;
}

function matrixSubtract(matrixA, matrixB){
	var matrixC = [];
	
	for(var i = 0; i < matrixA.length; i++){
		var column = [];
		for(var j = 0; j < matrixA[0].length; j++){
			column.push(matrixA[i][j]-matrixB[i][j]);
		}
		matrixC.push(column.slice(0));
	}
	
	return matrixC;
}

function matrixAdd(matrixA, matrixB){
	var matrixC = [];
	
	for(var i = 0; i < matrixA.length; i++){
		var column = [];
		for(var j = 0; j < matrixA[0].length; j++){
			column.push(matrixA[i][j]+matrixB[i][j]);
		}
		matrixC.push(column.slice(0));
	}
	
	return matrixC;
}

function hProduct(matrixA, matrixB){
	var matrixC = [];
	
	for(var i = 0; i < matrixA.length; i++){
		var column = [];
		for(var j = 0; j < matrixA[0].length; j++){
			column.push(matrixA[i][j]*matrixB[i][j]);
		}
		matrixC.push(column.slice(0));
	}
	
	return matrixC;
}

function costFromData(y, yHat){
	var cost = 0;
	for(var i = 0; i < y.length; y++){
		cost += 0.5*(y[i][0]-yHat[i][0][0])*(y[i][0]-yHat[i][0][0]);
	}
	return cost;
}

function errorFromData(y, yHat){
	var error = 0;
	for(var i = 0; i < y.length; i++){
		error += (yHat[i][0][0]-y[i][0]);
	}
	return error;
}

function flipMatrix(matrixA){
	var matrixC = [];
	
	for(var j = 0; j < matrixA[0].length; j++){
		var column = [];
		for(var i = 0; i < matrixA.length; i++){
			column.push(matrixA[i][j]);
		}
	
		matrixC.push(column);
	}
	return matrixC;
}

function sigmoid(t){
	return 1/(1+Math.pow(Math.E, -t));
}

function sigmoidPrime(t){
	return Math.pow(Math.E, -t)/Math.pow((1+Math.pow(Math.E, -t)), 2);
}

function arraySigmoid(matrixA){ //WARNING: only works on 1*n matrices
	var matrixC = [];
	for(var i = 0; i < matrixA.length; i++){
		matrixC.push([sigmoid(matrixA[i][0])]);
	}
	return matrixC;
}

function arraySigmoidPrime(matrixA){ //WARNING: only works on 1*n matrices
	var matrixC = [];
	for(var i = 0; i < matrixA.length; i++){
		matrixC.push([sigmoidPrime(matrixA[i][0])]);
	}
	return matrixC;
}

function dim(matrixA){
	return "[" + matrixA[0].length + " by " + matrixA.length + "]";
}