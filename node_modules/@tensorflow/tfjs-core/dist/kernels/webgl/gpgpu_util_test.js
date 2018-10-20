"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../../index");
var jasmine_util_1 = require("../../jasmine_util");
var test_util_1 = require("../../test_util");
var gpgpu_context_1 = require("./gpgpu_context");
var gpgpu_util = require("./gpgpu_util");
var DOWNLOAD_FLOAT_ENVS = {
    'WEBGL_DOWNLOAD_FLOAT_ENABLED': true
};
jasmine_util_1.describeWithFlags('gpgpu_util createWebGLContext', test_util_1.WEBGL_ENVS, function () {
    var gpgpu;
    beforeEach(function () {
        gpgpu = new gpgpu_context_1.GPGPUContext();
    });
    afterEach(function () {
        gpgpu.dispose();
    });
    it('disables DEPTH_TEST and STENCIL_TEST', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.DEPTH_TEST)).toEqual(false);
        expect(gpgpu.gl.getParameter(gpgpu.gl.STENCIL_TEST)).toEqual(false);
    });
    it('disables BLEND', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.BLEND)).toEqual(false);
    });
    it('disables DITHER, POLYGON_OFFSET_FILL', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.DITHER)).toEqual(false);
        expect(gpgpu.gl.getParameter(gpgpu.gl.POLYGON_OFFSET_FILL)).toEqual(false);
    });
    it('enables CULL_FACE with BACK', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE)).toEqual(true);
        expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE_MODE))
            .toEqual(gpgpu.gl.BACK);
    });
    it('enables SCISSOR_TEST', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_TEST)).toEqual(true);
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util createFloat32MatrixTexture', test_util_1.WEBGL_ENVS, function () {
    it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
    it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util createPackedMatrixTexture', test_util_1.WEBGL_ENVS, function () {
    it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
    it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util downloadMatrixFromPackedOutputTexture', DOWNLOAD_FLOAT_ENVS, function () {
    it('should work when texture shape != logical shape', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 4, 6, textureConfig);
        var mat = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12]);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        gpgpu.gl.texSubImage2D(gpgpu.gl.TEXTURE_2D, 0, 0, 0, 3, 2, gpgpu.gl.RGBA, gpgpu.gl.FLOAT, new Float32Array([
            0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0,
            6, 7, 0, 0, 8, 9, 0, 0, 10, 11, 0, 0
        ]));
        var result = gpgpu.downloadMatrixFromPackedTexture(tex, mat.shape, 4, 6);
        test_util_1.expectArraysClose(result, mat.dataSync());
    });
});
//# sourceMappingURL=gpgpu_util_test.js.map