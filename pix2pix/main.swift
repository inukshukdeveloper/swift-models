// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import Foundation
import Datasets
import ModelSupport
import pix2pix
import Checkpoints

let options = Options.parseOrExit()

let dataset = try! Pix2PixDataset(
    from: options.datasetPath,
    trainBatchSize: 1,
    testBatchSize: 1)

let validationImage = dataset.testSamples[0].source.expandingShape(at: 0)
let validationImage2 = dataset.testSamples[0].target.expandingShape(at: 0)

var validationImageURL = URL(string: FileManager.default.currentDirectoryPath)!.appendingPathComponent("sample.jpg")
var recreatedValidationImageURL = URL(string: FileManager.default.currentDirectoryPath)!.appendingPathComponent("recreated_sample.jpg")

var generator = NetG(inputChannels: 1, outputChannels: 1, ngf: 64, useDropout: false)
var discriminator = NetD(inChannels: 1, lastConvFilters: 64)
var discriminator2 = NetD(inChannels: 1, lastConvFilters: 64)

let optimizerG = Adam(for: generator, learningRate: 0.0002, beta1: 0.5)
let optimizerD = Adam(for: discriminator, learningRate: 0.0002, beta1: 0.5)
let optimizerD2 = Adam(for: discriminator2, learningRate: 0.0002, beta1: 0.5)


let epochCount = options.epochs
var step = 0
let lambdaL1 = Tensor<Float>(100)

private func computeY(r: UInt16, g: UInt16, b: UInt16) -> UInt16 {
    let rShift1 = UInt16(r << 6)
    let rShift2 = UInt16(r << 1)
    let gShift1 = UInt16(g << 7)
    let bShift1 = UInt16(b << 4)
    let bShift2 = UInt16(b << 3)
    let val1 = UInt16(UInt16(rShift1) + UInt16(rShift2))
    let shiftVals = UInt16(UInt16(rShift1) + UInt16(rShift2) + UInt16(gShift1) + UInt16(g) + UInt16(bShift1) + UInt16(bShift2) + UInt16(b))
    let val:UInt16 = UInt16(16 + (shiftVals >> 8))
    
//    let term1 = (65.738*Double(r))/256.0
//    let term2 = (129.057*Double(g))/256.0
//    let term3 = (25.064*Double(b))/256.0
//    let bruteY = round(16 + term1 + term2 + term3)
    
//    let shiftY = 16 + (((r << 6) + (r << 1) + (g << 7) + g + (b << 4) + (b << 3) + b) >> 8)
    return val
}

private func computeCb(r: UInt16, g: UInt16, b: UInt16) -> UInt16 {
    let bShift = Int16(b << 4)
    let gShift1 = g << 1
    let gShift3 = g << 3
    let gShift6 = g << 6
    let exp1 = Int16(gShift6 + gShift3 + gShift1)
    let rShift1 = r << 1
    let rShift2 = r << 2
    let rShift5 = r << 5
    let exp2 = Int16(rShift5 + rShift2 + rShift1)
    let exp = (-exp2 - exp1 + Int16(b << 7) - bShift)
    let val = UInt16(128 + (exp >> 8))
    
//    let Cb = round(128 - (37.945*Double(r))/256.0 - (74.494*Double(g))/256.0 + (112.439*Double(b))/256.0)
//    return UInt16(Cb)
    return val
}

private func computeCr(r: UInt16, g: UInt16, b: UInt16) -> UInt16 {
    let bShift:Int32 = Int32(((b << 4) + (b << 1)))
    let rShift7:Int32 = Int32(r << 7)
    let rShift4:Int32 = Int32(r << 4)
    let gShift6:Int32 = Int32(g << 6)
    let gShift5:Int32 = Int32(g << 5)
    let gShift1:Int32 = Int32(g << 1)
    let exp:Int32 = Int32((rShift7 - rShift4) - ((gShift6 + gShift5 - gShift1) - bShift))
    let val = UInt16(128 + exp >> 8)
    
//    let Cr = round(128 + (112.439*Double(r))/256.0 - (94.154*Double(g))/256.0 - (18.285*Double(b))/256.0)
//    return UInt16(Cr)
    return val
}

private func convertYUV2RGB(y: Float, u: Float, v: Float) -> [Float] {
//    let Y:Double = Double(y-16)
//    let U:Double = Double(u-128)
//    let V:Double = Double(v-128)
//    let r = 1.164*Double(Y)                   + 1.596*Double(V)
//    let g = 1.164*Double(Y) - 0.392*Double(U) - 0.813*Double(V)
//    let b = 1.164*Double(Y) + 2.017*Double(U)
//    return [UInt16(r),UInt16(g),UInt16(b)]
    
//    let Cr = v - 128
//    let Cb = u - 128
    
//    let r = y + ( Cr >> 2 + Cr >> 3 + Cr >> 5 )
//    let g = y - ( Cb >> 2 + Cb >> 4 + Cb >> 5) - ( Cr >> 1 + Cr >> 3 + Cr >> 4 + Cr >> 5)
//    let b = y + ( Cb + Cb >> 1 + Cb >> 2 + Cb >> 6)
    
//    let CrShift2 = Cr >> 2
//    let CrShift3 = Cr >> 3
//    let CrShift5 = Cr >> 5
//    let CbShift2 = Cb >> 2
//    let CbShift4 = Cb >> 4
//    let CbShift5 = Cb >> 5
//    let CrShift1 = Cr >> 1
//    let CrShift4 = Cr >> 4
//    let CbShift1 = Cb >> 1
//    let CbShift6 = Cb >> 6
//
//    var r = y + ( CrShift2 + CrShift3 + CrShift5 )
//    var g = y - ( CbShift2 + CbShift4 + CbShift5) - ( CrShift1 + CrShift3 + CrShift4 + CrShift5)
//    var b = y + ( Cb + CbShift1 + CbShift2 + CbShift6)
    
    // brute force  https://www.mir.com/DMG/ycbcr.html  rgb normalized to [0,1]
    // y, u, and v normalized
    let r2 = y + 1.402*u
    let g2 = y - 0.344136*u - 0.714136*v
    let b2 = y + 1.772*u

    return [r2, g2, b2]
}

private func convertYCbCrToRGB(yuv: Tensor<Float>) -> Tensor<Float> {
    var rgb = Tensor<Float>(repeating: 0, shape: yuv.shape)
    let Y = yuv.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
    let Cb = yuv.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
    let Cr = yuv.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])
    
    let squeezeY = Y.squeezingShape(at: 0,3)
    let squeezeCb = Cb.squeezingShape(at: 0,3)
    let squeezeCr = Cr.squeezingShape(at: 0,3)
    
    let yScalars = squeezeY.scalars
    let cbScalars = squeezeCb.scalars
    let crScalars = squeezeCr.scalars

    
    
    
    for i in 0..<256 {
        for j in 0..<256 {
            let yVal = yScalars[i*j]
            let cbVal = cbScalars[i*j]
            let crVal = crScalars[i*j]
            
//            let yFloat = Float(yVal)!
//            let yInt = Int(yFloat)
//            let cbFloat = Float(cbVal)!
//            let cbInt = Int(cbFloat)
//            let crFloat = Float(crVal)!
//            let crInt = Int(crFloat)
            
            
//            let y = yVal[0]
//            let cb = cbVal[0]
//            let cr = crVal[0]
            

//            var imageUnScaled = (rVal + 1) * 127.5
//            let rInt = UInt16(imageUnScaled)
//
//            imageUnScaled = (gVal + 1) * 127.5
//            let gInt = UInt16(imageUnScaled)
//
//            imageUnScaled = (bVal + 1) * 127.5
//            let bInt = UInt16(imageUnScaled)
            let rgbTmp = convertYUV2RGB(y: yVal, u: cbVal, v: crVal)
            let r = rgbTmp[0]
            let g = rgbTmp[1]
            let b = rgbTmp[2]
//
            rgb[0,i,j,0] = Tensor<Float>(Float(r))
            rgb[0,i,j,1] = Tensor<Float>(Float(g))
            rgb[0,i,j,2] = Tensor<Float>(Float(b))

//            YCbCr[0,i,j,1] = Float(Cb)
//            YCbCr[0,i,j,2] = Float(Cr)
//            print("unscaled", UInt8(imageUnScaled))
//            print("rVal", rVal)
//            print("rVal unsigned", rValUint)
        }
    }

    return rgb
}
/*
void RGBfromYUV(double& R, double& G, double& B, double Y, double U, double V)
{
  Y -= 16;
  U -= 128;
  V -= 128;
  R = 1.164 * Y             + 1.596 * V;
  G = 1.164 * Y - 0.392 * U - 0.813 * V;
  B = 1.164 * Y + 2.017 * U;
}
*/
private func convertRGB2YCbCr(rgb: Tensor<Float>) -> Tensor<Float> {
    
    var YCbCr = Tensor<Float>(repeating: 0, shape: rgb.shape)
    let t = Tensor<Int32>([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]])
//    print("YCbCr shape", YCbCr.shape)
    let tSlice = t.slice(lowerBounds: [2, 0, 0], sizes:[1, 1, 3])
//    print("tSlice", tSlice)
//    print("tSlice shape", tSlice.shape)
    let r = rgb.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
    let g = rgb.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
    let b = rgb.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])
    
 
//    print("shape of r", r.shape)
//    print("shape of g", g.shape)
//    print("shape of b", b.shape)
    
    // TODO: make this a mapping function where RGB => YCbCr
    for i in 0..<256 {
        for j in 0..<256 {
            let rVal:Float = Float(r[0,i,j,0])!
            let gVal:Float = Float(g[0,i,j,0])!
            let bVal:Float = Float(b[0,i,j,0])!

            var imageUnScaled = (rVal + 1) * 127.5
            let rInt = UInt16(imageUnScaled)

            imageUnScaled = (gVal + 1) * 127.5
            let gInt = UInt16(imageUnScaled)
            
            imageUnScaled = (bVal + 1) * 127.5
            let bInt = UInt16(imageUnScaled)
            
            var Y = computeY(r: rInt, g: gInt, b: bInt)
            var Cb = computeCb(r: rInt, g: gInt, b: bInt)
            var Cr = computeCr(r: rInt, g: gInt, b: bInt)
            
            // regularize back the components to -1,1
//            let Y2 = Double(Y) / 127.5 - 1.0
//            let Cb2 = Double(Cb) / 127.5 - 1.0
//            let Cr2 = Double(Cr) / 127.5 - 1.0
            
            // [0,1]  https://www.mir.com/DMG/ycbcr.html
//            let Y2 = (Double(Y) - 16.0) / 219.0  // [0,1]
//            let Cb2 = (Double(Cb) - 128.0) / 224.0  // [-.5,.5]
//            let Cr2 = (Double(Cr) - 128.0) / 224.0  // [-.5,.5]

            
            YCbCr[0,i,j,0] = Tensor<Float>(Float(Y))
            YCbCr[0,i,j,1] = Tensor<Float>(Float(Cb))
            YCbCr[0,i,j,2] = Tensor<Float>(Float(Cr))

//            YCbCr[0,i,j,1] = Float(Cb)
//            YCbCr[0,i,j,2] = Float(Cr)
//            print("unscaled", UInt8(imageUnScaled))
//            print("rVal", rVal)
//            print("rVal unsigned", rValUint)
        }
    }

//    let rg = r.concatenated(with: g, alongAxis: 3)
    
//    let rgb = rg.concatenated(with: b, alongAxis: 3)
    
    
    
    return YCbCr
}

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() {
    print("Epoch \(epoch) started at: \(Date())")
    
    
    var discriminatorTotalLoss = Tensor<Float>(0)
    var discriminatorTotalLoss2 = Tensor<Float>(0)

    var generatorTotalLoss = Tensor<Float>(0)
    var discriminatorCount = 0
    
    for batch in epochBatches {
        print("Batch \(step) started at \(Date())")
        defer { step += 1 }

        Context.local.learningPhase = .training
//        print("shape of source", batch.source.shape)
//        print("shape of target", batch.target.shape)
        
//        let yccImageSource = convertRGB2YCbCr(rgb: batch.source)
//        let yccImageTarget = convertRGB2YCbCr(rgb: batch.target)
        
        let yccImageSource = batch.source  // should already be converted and normalized
        let yccImageTarget = batch.target  // should already be converted and normalized
        
        let fused = yccImageSource + yccImageTarget
//        print("shape of fused", fused.shape)
        
//        let sourceY = yccImageSource[1]
        
        let concatanatedImages = batch.source.concatenated(with: batch.target)
        
        let concatenatedYCbCrImages = yccImageSource.concatenated(with: yccImageTarget)
//        print("concatenated shape", concatanatedImages.shape)
//        print("YCbCr shape", concatenatedYCbCrImages.shape)
                         
        let scaledImages = _Raw.resizeNearestNeighbor(images: concatanatedImages, size: [286, 286])
        var croppedImages = scaledImages.slice(lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
                                               sizes: [2, 256, 256, 3])
        
        let scaledYCbCrImages = _Raw.resizeNearestNeighbor(images: concatenatedYCbCrImages, size: [286, 286])
        var croppedYCbCrImages = scaledYCbCrImages.slice(lowerBounds: Tensor<Int32>([0, Int32(random() % 30), Int32(random() % 30), 0]),
                                                    sizes: [2, 256, 256, 3])
        
        if Bool.random() {
            croppedImages = _Raw.reverse(croppedImages, dims: [false, false, true, false])
            croppedYCbCrImages = _Raw.reverse(croppedYCbCrImages, dims: [false, false, true, false])

        }
        
        let sourceImages = croppedImages[0].expandingShape(at: 0)
        let targetImages = croppedImages[1].expandingShape(at: 0)
        
//        print("source images shape", sourceImages.shape)
//        print("target images shape", targetImages.shape)
        
        let sourceYCbCrImages = croppedYCbCrImages[0].expandingShape(at: 0)
        let targetYCbCrImages = croppedYCbCrImages[1].expandingShape(at: 0)
        
        let ySource = sourceYCbCrImages.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
        let yTarget = targetYCbCrImages.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
        
          
        let cbSource = sourceYCbCrImages.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
        let cbTarget = targetYCbCrImages.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
        
        let crSource = sourceYCbCrImages.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])
        let crTarget = targetYCbCrImages.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])

//        print("ySource shape", ySource.shape)
//        print("yTarget shape", yTarget.shape)
        
        let fusedImages = sourceImages+targetImages
        
//        print("shape of fused images", fusedImages.shape)
        

 
        // present fused image to generator.  Source will be either sat or hillshade.  Target will
        // be the other.

        let generatorGradient = TensorFlow.gradient(at: generator) { g -> Tensor<Float> in
            
            
            let fakeImages = g(ySource+yTarget)
//            print("fake images shape", fakeImages.shape)
//            let fakeAB = sourceImages.concatenated(with: fakeImages, alongAxis: 3)
            let fakeAB = fakeImages-yTarget
            let fakeAB2 = fakeImages-ySource
            
            
            // F-S2 presented to Discrimator1 and F-S1 presented to Discriminator2
            
            let fakePrediction = discriminator(fakeAB)
            let fakePrediction2 = discriminator2(fakeAB2)
                        
            let ganLoss = sigmoidCrossEntropy(logits: fakePrediction,
                                              labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            let ganLoss2 = sigmoidCrossEntropy(logits: fakePrediction2,
                                               labels: Tensor<Float>.one.broadcasted(to: fakePrediction2.shape))
//            let l1Loss = meanAbsoluteError(predicted: fakeImages,
//                                           expected: targetImages) * lambdaL1
            
            let l1Loss = meanAbsoluteError(predicted: fakeImages,
                                            expected: yTarget+ySource) * lambdaL1

            generatorTotalLoss += ganLoss + ganLoss2  + l1Loss
            return ganLoss + ganLoss2 + l1Loss
        }
        
        let fakeImages = generator(ySource+yTarget)
                
        
        // cmopute each discriminator loss from fakes and reals
        let descriminatorGradient = TensorFlow.gradient(at: discriminator) { d -> Tensor<Float> in
//            let fakeAB = sourceImages.concatenated(with: fakeImages,
//                                                   alongAxis: 3)
            let fakeAB2 = fakeImages-yTarget
            
            let fakePrediction = d(fakeAB2)
//            print("fake prediction shape", fakePrediction.shape)
            let fakeLoss = sigmoidCrossEntropy(logits: fakePrediction,
                                               labels: Tensor<Float>.zero.broadcasted(to: fakePrediction.shape))
            
//            let realAB = sourceImages.concatenated(with: targetImages,
//                                                   alongAxis: 3)
            let realAB2 = yTarget
            let realPrediction = d(realAB2)
            let realLoss = sigmoidCrossEntropy(logits: realPrediction,
                                               labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            
            discriminatorTotalLoss += (fakeLoss + realLoss) * 0.5
            return (fakeLoss + realLoss) * 0.5
        }
        
        let descriminatorGradient2 = TensorFlow.gradient(at: discriminator2) { d -> Tensor<Float> in
//            let fakeAB = sourceImages.concatenated(with: fakeImages,
//                                                   alongAxis: 3)
            let fakeAB2 = fakeImages-ySource
            let fakePrediction = d(fakeAB2)
//            print("fake prediction shape D2", fakePrediction.shape)

            let fakeLoss = sigmoidCrossEntropy(logits: fakePrediction,
                                               labels: Tensor<Float>.zero.broadcasted(to: fakePrediction.shape))
            
//            let realAB = sourceImages.concatenated(with: targetImages,
//                                                   alongAxis: 3)
            let realAB2 = ySource
            let realPrediction = d(realAB2)
            let realLoss = sigmoidCrossEntropy(logits: realPrediction,
                                               labels: Tensor<Float>.one.broadcasted(to: fakePrediction.shape))
            
            discriminatorTotalLoss2 += (fakeLoss + realLoss) * 0.5
            return (fakeLoss + realLoss) * 0.5
        }

        
        optimizerG.update(&generator, along: generatorGradient)
        optimizerD.update(&discriminator, along: descriminatorGradient)
        optimizerD2.update(&discriminator2, along: descriminatorGradient2)
                
        // MARK: Sample Inference
        if step % options.sampleLogPeriod == 0 {
            Context.local.learningPhase = .inference
                        
//            let fakeSample = generator(validationImage) * 0.5 + 0.5
            let yccImageSource = convertRGB2YCbCr(rgb: validationImage)
            let yccImageTarget = convertRGB2YCbCr(rgb: validationImage2)

            let validationImageY = yccImageSource.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
            let validationImage2Y = yccImageTarget.slice(lowerBounds: [0,0,0,0], sizes: [1,256,256,1])
            
            let valImageCb = yccImageSource.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
            let valImageCr = yccImageSource.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])

            let valImage2Cb = yccImageTarget.slice(lowerBounds: [0,0,0,1], sizes: [1,256,256,1])
            let valImage2Cr = yccImageTarget.slice(lowerBounds: [0,0,0,2], sizes: [1,256,256,1])

            // present source + target validation image to generator
            var fakeSample2 = generator(validationImageY+validationImage2Y) * 0.5 + 0.5
            
            // add other components back to Y
            fakeSample2 = fakeSample2.concatenated(with: valImageCb+valImage2Cb, alongAxis: 3)
            fakeSample2 = fakeSample2.concatenated(with: valImageCr+valImage2Cr, alongAxis: 3)
            
//            print("shape of fakeSample2", fakeSample2.shape)
            
            
            // convert back to RGB and save
//            print("fakeSample2", fakeSample2)
            let rgbImage = convertYCbCrToRGB(yuv: fakeSample2)

//            let fakeSampleImage = Image(tensor: fakeSample[0] * 255)
//            print("rgbImage", rgbImage)

            let fakeSampleImage2 = Image(tensor: rgbImage[0] * 255)
            let name = "sample" + String(epoch) + String(step) + ".jpg"
            validationImageURL = URL(string: FileManager.default.currentDirectoryPath)!.appendingPathComponent(name)
            fakeSampleImage2.save(to: validationImageURL, format: .rgb)
        }
        discriminatorCount += 1
    }
    
    let generatorLoss = generatorTotalLoss / Float(discriminatorCount)
    let discriminatorLoss = discriminatorTotalLoss / Float(discriminatorCount)
    let discriminatorLoss2 = discriminatorTotalLoss2 / Float(discriminatorCount)
    print("Generator train loss: \(generatorLoss.scalars[0])")
    print("Discriminator train loss: \(discriminatorLoss.scalars[0])")
    print("Discriminator2 train loss: \(discriminatorLoss2.scalars[0])")

    
    
    // now combine the Cb and Cr components back into the final
    // fused image
}

Context.local.learningPhase = .inference

var totalLoss = Tensor<Float>(0)
var count = 0

let resultsFolder = try createDirectoryIfNeeded(path: FileManager.default.currentDirectoryPath + "/results")
for batch in dataset.testing {
    let fakeImages = generator(batch.source)

    let tensorImage = batch.source
                           .concatenated(with: fakeImages,
                                         alongAxis: 2) / 2.0 + 0.5

    let image = Image(tensor: (tensorImage * 255)[0])
    let saveURL = resultsFolder.appendingPathComponent("\(count).jpg", isDirectory: false)
    image.save(to: saveURL, format: .rgb)

    let ganLoss = sigmoidCrossEntropy(logits: fakeImages,
                                      labels: Tensor.one.broadcasted(to: fakeImages.shape))
    let l1Loss = meanAbsoluteError(predicted: fakeImages,
                                   expected: batch.target) * lambdaL1

    totalLoss += ganLoss + l1Loss
    count += 1
}

let testLoss = totalLoss / Float(count)
print("Generator test loss: \(testLoss.scalars[0])")
