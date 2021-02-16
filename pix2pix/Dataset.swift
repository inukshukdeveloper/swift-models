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
import Foundation
import ModelSupport
import Datasets
import TensorFlow

public enum Pix2PixDatasetVariant: String {
    case facades
    case maps

    public var url: URL {
        switch self {
        case .facades:
            return URL(string:
                "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip")!
        case .maps:
            return URL(string:
                "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip")!
        }
    }
}

public struct Pix2PixDataset<Entropy: RandomNumberGenerator> {
    public typealias Samples = [(source: Tensor<Float>, target: Tensor<Float>)]
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    public typealias PairedImageBatch = (source: Tensor<Float>, target: Tensor<Float>)
    public typealias Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>,
        LazyMapSequence<Batches, PairedImageBatch>
      >
    public typealias Testing = LazyMapSequence<
        Slices<Samples>,
        PairedImageBatch
    >

    public let trainSamples: Samples
    public let testSamples: Samples
    public let training: Training
    public let testing: Testing

    public init(
        from rootDirPath: String? = nil,
        variant: Pix2PixDatasetVariant? = nil,
        trainBatchSize: Int = 1,
        testBatchSize: Int = 1,
        entropy: Entropy) throws {
        
        let rootDirPath = rootDirPath ?? Pix2PixDataset.downloadIfNotPresent(
            variant: variant ?? .maps,
            to: DatasetUtilities.defaultDirectory.appendingPathComponent("pix2pix", isDirectory: true))
        let rootDirURL = URL(fileURLWithPath: rootDirPath, isDirectory: true)
        
        trainSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("trainA"),
                  fileIndexRetriever: "_"
                ),
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("trainB"),
                  fileIndexRetriever: "_"
                )
        ))
        testSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("testA"),
                  fileIndexRetriever: "_"
                ),
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL.appendingPathComponent("testB"),
                  fileIndexRetriever: "_"
                )
        ))
        training = TrainingEpochs(
            samples: trainSamples,
            batchSize: trainBatchSize,
            entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, PairedImageBatch> in
            batches.lazy.map {
                (
                    source: Tensor<Float>($0.map(\.source)),
                    target: Tensor<Float>($0.map(\.target))
                )
            }
        }

        testing = testSamples.inBatches(of: testBatchSize)
            .lazy.map {
                (
                    source: Tensor<Float>($0.map(\.source)),
                    target: Tensor<Float>($0.map(\.target))
                )
            }
    }

    private static func downloadIfNotPresent(
            variant: Pix2PixDatasetVariant,
            to directory: URL) -> String {
        let rootDirPath = directory.appendingPathComponent(variant.rawValue).path

        let directoryExists = FileManager.default.fileExists(atPath: rootDirPath)
        let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: rootDirPath)
        let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)
        guard !directoryExists || directoryEmpty else { return rootDirPath }

        let _ = DatasetUtilities.downloadResource(
            filename: variant.rawValue,
            fileExtension: "zip",
            remoteRoot: variant.url.deletingLastPathComponent(),
            localStorageDirectory: directory)
        print("\(rootDirPath) downloaded.")

        return rootDirPath
    }

    
    private static func computeY(r: Int, g: Int, b: Int) -> Int {
        let rShift1 = r << 6
        let rShift2 = r << 1
        let gShift1 = g << 7
        let bShift1 = b << 4
        let bShift2 = b << 3
//        let val1 = UInt16(UInt16(rShift1) + UInt16(rShift2))
        let shiftVals = ((rShift1) + (rShift2) + (gShift1) + (g) + (bShift1) + (bShift2) + (b))
        let val = (16 + (shiftVals >> 8))
        
    //    let term1 = (65.738*Double(r))/256.0
    //    let term2 = (129.057*Double(g))/256.0
    //    let term3 = (25.064*Double(b))/256.0
    //    let bruteY = round(16 + term1 + term2 + term3)
        
    //    let shiftY = 16 + (((r << 6) + (r << 1) + (g << 7) + g + (b << 4) + (b << 3) + b) >> 8)
        return val
    }

    private static func computeCb(r: Int, g: Int, b: Int) -> Int {
        let bShift = (b << 4)
        let gShift1 = g << 1
        let gShift3 = g << 3
        let gShift6 = g << 6
        let exp1 = gShift6 + gShift3 + gShift1
        let rShift1 = r << 1
        let rShift2 = r << 2
        let rShift5 = r << 5
        let bShift7 = b << 7
        let exp2 = rShift5 + rShift2 + rShift1
        let exp = (-exp2 - exp1) + bShift7 - bShift
        let val = 128 + (exp >> 8)
        
    //    let Cb = round(128 - (37.945*Double(r))/256.0 - (74.494*Double(g))/256.0 + (112.439*Double(b))/256.0)
    //    return UInt16(Cb)
        return val
    }

    private static func computeCr(r: Int, g: Int, b: Int) -> Int {
        let bShift = (b << 4) + (b << 1)
        let rShift7 = r << 7
        let rShift4 = r << 4
        let gShift6 = g << 6
        let gShift5 = g << 5
        let gShift1 = g << 1
        let exp = (rShift7 - rShift4) - ((gShift6 + gShift5 - gShift1) - bShift)
        let val = 128 + (exp >> 8)
        
    //    let Cr = round(128 + (112.439*Double(r))/256.0 - (94.154*Double(g))/256.0 - (18.285*Double(b))/256.0)
    //    return UInt16(Cr)
        return val
    }

    private static func convertYUV2RGB(y: Float, u: Float, v: Float) -> [Float] {
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
        let r2 = y + 1.402*u
        let g2 = y - 0.344136*u - 0.714136*v
        let b2 = y + 1.772*u

        return [r2, g2, b2]
    }
    
    private static func convertRGB2YCbCr(rgb: Tensor<Float>) -> Tensor<Float> {
        
        var YCbCr = Tensor<Float>(repeating: 0, shape: rgb.shape)
        var r = rgb.slice(lowerBounds: [0,0,0], sizes: [256,256,1])
        var g = rgb.slice(lowerBounds: [0,0,1], sizes: [256,256,1])
        var b = rgb.slice(lowerBounds: [0,0,2], sizes: [256,256,1])
        
        r = r.squeezingShape(at: 2)
        g = g.squeezingShape(at: 2)
        b = b.squeezingShape(at: 2)
        
        let rScalars = r.scalars
        let gScalars = g.scalars
        let bScalars = b.scalars
        
        var rInt = 0
        
//        let new = _Raw.mapDataset(inputDataset: <#T##VariantHandle#>, otherArguments: <#T##TensorArrayProtocol#>, f: <#T##(TensorGroup) -> TensorGroup#>, outputTypes: <#T##[TensorDataType]#>, outputShapes: <#T##[TensorShape?]#>, useInterOpParallelism: <#T##Bool#>, preserveCardinality: <#T##Bool#>)
                
        // TODO: make this a mapping function where RGB => YCbCr
        for i in 0..<256 {
            for j in 0..<256 {
                let rVal = Int(rScalars[i*j])
                let gVal = Int(gScalars[i*j])
                let bVal = Int(bScalars[i*j])
                
//                let val2:Float = val.scalar!
//                let val3:Int = Int(val2)
//                let gVal = g[i,j]
//                let bVal = b[i,j]
                
                var Y = computeY(r: rVal, g: gVal, b: bVal)
                var Cb = computeCb(r: rVal, g: gVal, b: bVal)
                var Cr = computeCr(r: rVal, g: gVal, b: bVal)
                  
                
//                var Y = computeY(r: 0, g: 1, b: 2)
//                var Cb = computeCb(r: 0, g: 1, b: 2)

                // [0,1]  https://www.mir.com/DMG/ycbcr.html
//                let Y2 = (Double(Y) - 16.0) / 219.0  // [0,1]
//                let Cb2 = (Double(Cb) - 128.0) / 224.0  // [-.5,.5]
//                let Cr2 = (Double(Cr) - 128.0) / 224.0  // [-.5,.5]

//                YCbCr[i,j,0] = Tensor<Float>(Float(Y2))
//                YCbCr[i,j,1] = Tensor<Float>(Float(Cb2))
//                YCbCr[i,j,2] = Tensor<Float>(Float(Cr2))
                
//                YCbCr[i,j,0] = Tensor<Float>(0.0)
//                YCbCr[i,j,1] = Tensor<Float>(1.0)
//                YCbCr[i,j,2] = Tensor<Float>(2.0)

            }
        }
        return YCbCr
    }

//    private static func convertYCbCrToRGB(yuv: Tensor<Float>) -> Tensor<Float> {
//        var rgb = Tensor<Float>(repeating: 0, shape: yuv.shape)
//        let Y = yuv.slice(lowerBounds: [0,0,0], sizes: [256,256,1])
//        let Cb = yuv.slice(lowerBounds: [0,0,1], sizes: [256,256,1])
//        let Cr = yuv.slice(lowerBounds: [0,0,2], sizes: [256,256,1])
//
//        let squeezeY = Y.squeezingShape(at: 2)
//        let squeezeCb = Cb.squeezingShape(at: 2)
//        let squeezeCr = Cr.squeezingShape(at: 2)
//
//
//
//
//        for i in 0..<256 {
//            for j in 0..<256 {
//                let yVal = squeezeY[i,j]
//                let cbVal = squeezeCb[i,j]
//                let crVal = squeezeCr[i,j]
//
//                let yFloat = Float(yVal)!
//                let yInt = Int(yFloat)
//                let cbFloat = Float(cbVal)!
//                let cbInt = Int(cbFloat)
//                let crFloat = Float(crVal)!
//                let crInt = Int(crFloat)
//
//
//    //            let y = yVal[0]
//    //            let cb = cbVal[0]
//    //            let cr = crVal[0]
//
//
//    //            var imageUnScaled = (rVal + 1) * 127.5
//    //            let rInt = UInt16(imageUnScaled)
//    //
//    //            imageUnScaled = (gVal + 1) * 127.5
//    //            let gInt = UInt16(imageUnScaled)
//    //
//    //            imageUnScaled = (bVal + 1) * 127.5
//    //            let bInt = UInt16(imageUnScaled)
//                let rgbTmp = convertYUV2RGB(y: yFloat, u: cbFloat, v: crFloat)
//                let r = rgbTmp[0]
//                let g = rgbTmp[1]
//                let b = rgbTmp[2]
//    //
//                rgb[0,i,j,0] = Tensor<Float>(Float(r))
//                rgb[0,i,j,1] = Tensor<Float>(Float(g))
//                rgb[0,i,j,2] = Tensor<Float>(Float(b))
//
//    //            YCbCr[0,i,j,1] = Float(Cb)
//    //            YCbCr[0,i,j,2] = Float(Cr)
//    //            print("unscaled", UInt8(imageUnScaled))
//    //            print("rVal", rVal)
//    //            print("rVal unsigned", rValUint)
//            }
//        }
//
//        return rgb
//    }
    
    
    private static func loadSortedSamples(
        from directory: URL,
        fileIndexRetriever: String
    ) throws -> [Tensor<Float>] {
        return try FileManager.default
            .contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "jpg" }
            .sorted {
                Int($0.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])! <
                Int($1.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])!
            }
            .map {
                var image = Image(jpeg: $0)
                image = image.resized(to: (256,256))
//                let imageScaled = image.tensor / 127.5 - 1.0
                let imageScaled = convertRGB2YCbCr(rgb: image.tensor)
                return imageScaled
            }
    }
}

extension Pix2PixDataset where Entropy == SystemRandomNumberGenerator {
    public init(
        from rootDirPath: String? = nil,
        variant: Pix2PixDatasetVariant? = nil,
        trainBatchSize: Int = 1,
        testBatchSize: Int = 1
    ) throws {
        try self.init(
            from: rootDirPath,
            variant: variant,
            trainBatchSize: trainBatchSize,
            testBatchSize: testBatchSize,
            entropy: SystemRandomNumberGenerator()
        )
    }
}
