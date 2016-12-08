
var myCanvas = document.getElementById("myCanvas");
var ctx = myCanvas.getContext("2d");

function graphNN(step){
	var count = Math.floor(1/step);
	var percentage = 0;
	
	//x goes from 50 to 450
	//y goes from 450 to 50
	
	ctx.clearRect(0, 0, myCanvas.width, myCanvas.height);
	
	for(var i = 0; i < data.length; i++){
		ctx.fillStyle = "#00FF00";
		ctx.fillRect(50+data[i][0]*400, 450-400*data[i][1], 2, 2);
	}
	
	for(var i = 0; i < count; i++){
		percentage = i/count;
		
		myNN.sendInput([[percentage]]);
		myNN.forwardProp();
		
		
		var currX = 50+percentage*400;
		var currY = 450 - 400*myNN.nLayers[myNN.LAYERS_SIZE.length-1][0][0];
		
		ctx.fillStyle = "#0000FF";
		ctx.fillRect(50+percentage*400, 450-400*graph(percentage), 3, 3);
		
		ctx.fillStyle = "#FF0000";
		ctx.fillRect(currX, currY, 3, 3);
	}
	
	
}
//engine:

function graph(x){
	return Math.sin(Math.PI*x);
}

function train(times){
	for(var i = 0; i < times; i++){
		var yHat = [];
		myNN.resetGradients();
		for(var j = 0; j < 500; j++){
			myNN.sendInput([input_set[j]]);
			myNN.forwardProp();
			yHat.push(myNN.nLayers[myNN.LAYERS_SIZE.length-1]);
			myNN.backwardProp([answer_set[j]]);
		}
		console.log(costFromData(answer_set, yHat));
		myNN.adjust();
	}
}



var input_set = [];
var answer_set = [];

for(var i = 0; i < 500; i++){
	var curr_x = Math.random();
	input_set.push([curr_x]);
	answer_set.push([graph(curr_x)]);
}

myNN = new NN();
myNN.allocSpace();
myNN.resetGradients();


/*myNN = new NN();
var input = [];

for(var i = 0; i < 784; i++){ input.push([Math.random()]); }

myNN.sendInput(input);

myNN.allocSpace();

var y = [[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]];

myNN.forwardProp();*/