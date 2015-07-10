//
//  NeuralNetwork.swift
//  Brain
//
//  Created by Vishal on 2014-06-08.
//  Copyright (c) 2015 Vishal. All rights reserved.
//

import Foundation

infix operator ** { associativity left precedence 160 }
func ** (num: Double, power: Double) -> Double{
    return pow(num, power)
}

infix operator *& { associativity left precedence 160 }
func *& (fill: Array<Double>, I: NSInteger) -> Array<Double>{
    var m = Array<Double>()
    let length = fill.count-1
    for _ in 1...I {
        for index in 0...length {
            m.append(fill[index])
        }
    }
    
    return m
}

func randomFunc(a: Double, b:Double) -> (Double) {
    let randNum = arc4random_uniform(100)/100
    let output = (b-a)*Double(randNum) + (a)
    
    return output
}

func makeMatrix(I:NSInteger, J:NSInteger)->(Array<Array<Double>>){
    let NumColumns = I
    let NumRows = J
    var array = Array<Array<Double>>()
    for _ in 0...NumColumns-1 {
        array.append(Array(count:NumRows, repeatedValue:Double()))
    }
    
    return array
}

//sigmoid function. Later, will add more options for standard 1/(1+e^-x)
func sigmoid(x: Double)->(Double){
    return tanh(x)
}


// derivative of our sigmoid function
func dsigmoid(x: Double)->(Double){
    return 1.0 - x**2.0
}

class NN {
    
    // Using default values may break this... You should initialize ni,nh,no
    var ni = 2
    var nh = 2
    var no = 2
    var ai = Array<Double>()
    var ah = Array<Double>()
    var ao = Array<Double>()
    var wi = Array<Array<Double>>()
    var wo = Array<Array<Double>>()
    var ci = Array<Array<Double>>()
    var co = Array<Array<Double>>()
    
    init(ni:NSInteger, nh:NSInteger, no:NSInteger) {
        // number of input, hidden, and output nodes
        self.ni = ni + 1 // +1 for bias node
        self.ni = ni // +1 for bias node
        self.nh = nh
        self.no = no
        
        // activations for nodes
        self.ai = [1.0]*&self.ni
        self.ah = [1.0]*&self.nh
        self.ao = [1.0]*&self.no
        
        //        my breakpoint
        
        print("ai: \(self.ai)")
        print("ah: \(self.ah)")
        print("ao: \(self.ao)")
        
        //create weights
        self.wi = makeMatrix(self.ni, J: self.nh)
        self.wo = makeMatrix(self.nh, J: self.no)
        
        for i in 0...(self.ni-1){
            for j in 0...(self.nh-1){
                print("init \(self.wi[0][1])")
                
                self.wi[i][j]=randomFunc(-0.2, b: 0.2)
            }
        }
        
        for j in 0...(self.nh-1){
            for k in 0...(self.no-1){
                self.wo[j][k] = randomFunc(-2.0, b: 2.0)
            }
        }
        
        // last change in weights for momentum
        self.ci = makeMatrix(self.ni, J: self.nh)
        self.co = makeMatrix(self.nh, J: self.no)
        
    }
    
    func update(inputs:Array<Double>) -> (Array<Double>) {
        if (inputs.count != self.ni-1){
            print("wrong number of inputs ")
        }
        
        // input activations
        print(" ")
        print("inputs \(inputs) ")
        
        print("a inputs \(self.ai) ")
        print("w inputs \(self.wi) ")
        print("c inputs \(self.ci) ")
        
        print("a hidden \(self.ah) ")
        
        print("a output \(self.ao) ")
        print("w output \(self.wo) ")
        print("c output \(self.co) ")
        
        print(" ")
        
        //        var ai = Array<Double>()
        //        var ah = Array<Double>()
        //        var ao = Array<Double>()
        //        var wi = Array<Array<Double>>()
        //        var wo = Array<Array<Double>>()
        //        var ci = Array<Array<Double>>()
        //        var co = Array<Array<Double>>()
        
        
        print("set inputs ")
        for i in 0...(self.ni-1) {
            print("i \(i) ")
            
            //self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]
        }
        
        print("hidden activations")
        // hidden activations
        for j in 1...(self.nh-1) {
            print("j \(j)")
            
            var sum = 0.0
            //            for i in 1...(self.ni-1){
            for i in 0...(self.ni-1){
                sum = sum + self.ai[i] * self.wi[i][j]
            }
            
            self.ah[j] = sigmoid(sum)
            
        }
        
        print("output activations \(self.no-1) ")
        // output activations
        //        for k in 1...(self.no-1) {
        for k in 0...(self.no-1) {
            var sum = 0.0
            for j in 0...(self.nh-1){
                sum = sum + self.ah[j] * self.wo[j][k]
            }
            
            self.ao[k] = sigmoid(sum)
        }
        
        return self.ao
    }
    
    func backPropagate(targets:Array<Double>, N:Double, M:Double)->(Double){
        if targets.count != self.no{
            print("wrong number of target values ")
        }
        
        //        print("calculate output deltas")
        
        // calculate error terms for output
        var output_deltas = [0.0] *& self.no
        //        for k in 0...(self.no) {
        for k in 0...(self.no-1) {
            //            print(k)
            //            print(targets)
            //            print(output_deltas)
            //            print(self.ao)
            let error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        }
        
        //        print("calculate hidden deltas")
        
        // calculate error terms for hidden
        var hidden_deltas = [0.0] *& self.nh
        //        for j in 0...(self.nh){
        for j in 0...(self.nh-1){
            var error = 0.0
            //            for k in 0...(self.no){
            for k in 0...(self.no-1){
                error = error + output_deltas[k]*self.wo[j][k]
            }
            
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        }
        
        //        print("calculate change")
        
        // update output weights
        //        for j in 0...(self.nh){
        for j in 0...(self.nh-1){
            //            for k in 0...(self.no){
            for k in 0...(self.no-1){
                print("j k \(j) \(k)")
                
                let change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                print (N*change)
                print (M*self.co[j][k])
            }
        }
        
        //        print("update input weights")
        
        // update input weights
        //        for i in 0...(self.ni){
        for i in 0...(self.ni-1){
            //            for j in 0...(self.nh){
            for j in 0...(self.nh-1){
                let change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
            }
        }
        
        // calculate error
        var error = 0.0
        //        for k in 0...(targets.count){
        for k in 0...(targets.count-1){
            error = error + 0.5*(targets[k]-self.ao[k])**2
        }
        
        print("calculate error: \(error)")
        
        return error
    }
    
    func test(patterns:Array<Array<Array<Double>>>)->(){
        for p in 0...(patterns.count - 1) {
            print("patterns \(patterns[p][0]) ->  \(self.update(patterns[p][0]))")
        }
    }
    
    func weights()->(){
        print("Input weights: ")
        for i in 0...(self.ni-1){
            print(self.wi[i])
            print("Output weights: ")
        }
        
        for j in 0...(self.nh-1){
            print(self.wo[j])
        }
        
    }
    
    func train(patterns:Array<Array<Array<Double>>>, iterations:NSInteger=10, N:Double=0.5, M:Double=0.1){
        // N: learning rate
        // M: momentum factor
        for i in 0...iterations-1 {
            var error = 0.0
            for p in 0...patterns.count-1{
                let inputs = patterns[p][0]
                let targets = patterns[p][1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N: N, M: M)
            }
            
            if i % 100 == 0 {
                print("error \(error) ")
            }
        }
    }
}

func demo()->(){
    //Teach network XOR function
    var pat = Array<Array<Array<Double>>>()
    
    var test_pat = Array<Array<Array<Double>>>()
    
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
    
    test_pat = [
        [[0,0]],
        [[0,1]],
        [[1,0]],
        [[1,1]]
    ]
    
    // create a network with two input, two hidden, and one output nodes
    let n = NN(ni: 2,nh: 2,no: 1)
    // train it with some patterns
    n.train(pat)
    // test it
    n.test(test_pat)
}

// let myFirstNN = NN(ni: 10,nh: 10,no: 10)
// var x = [2.0]*&4
print(demo())


