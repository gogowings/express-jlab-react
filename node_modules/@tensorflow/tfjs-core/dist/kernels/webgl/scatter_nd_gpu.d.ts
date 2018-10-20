import { GPGPUProgram } from './gpgpu_math';
export declare class ScatterNDProgram implements GPGPUProgram {
    private updateSize;
    private sliceDim;
    private strides;
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(updateSize: number, sliceDim: number, strides: number[], shape: number[]);
}
