"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var shader_compiler_1 = require("./shader_compiler");
var ScatterNDProgram = (function () {
    function ScatterNDProgram(updateSize, sliceDim, strides, shape) {
        this.updateSize = updateSize;
        this.sliceDim = sliceDim;
        this.strides = strides;
        this.variableNames = ['updates', 'indices'];
        this.outputShape = shape;
        var stridesType = shader_compiler_1.getCoordsDataType(strides.length);
        var dtype = shader_compiler_1.getCoordsDataType(shape.length);
        var strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
        this.userCode = "\n        " + stridesType + " strides = " + stridesType + "(" + this.strides + ");\n\n        void main() {\n          " + dtype + " coords = getOutputCoords();\n          float sum = 0.0;\n          for (int i = 0; i < " + this.updateSize + "; i++) {\n            int flattenIndex = 0;\n            for (int j = 0; j < " + this.sliceDim + "; j++) {\n              int index = round(getIndices(i, j));\n              flattenIndex += index * " + strideString + ";\n            }\n            if (flattenIndex == coords[0]) {\n              sum += getUpdates(i, coords[1]);\n            }\n          }\n          setOutput(sum);\n        }\n      ";
    }
    return ScatterNDProgram;
}());
exports.ScatterNDProgram = ScatterNDProgram;
//# sourceMappingURL=scatter_nd_gpu.js.map