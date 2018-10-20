import { Tensor } from '../tensor';
export declare function validateUpdateShape(shape: number[], indices: Tensor, updates: Tensor): void;
export declare function prepareAndValidate(updates: Tensor, indices: Tensor, shape: number[]): [number, number, number, number[], number];
