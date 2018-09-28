import cv2
import json
import boto3
import base64
import logging
import numpy as np

from io import BytesIO
from functools import wraps
from datetime import date, datetime
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, Union, List

from minerva_lib import crop
from minerva_scripts.omeroapi import OmeroApi
from minerva_scripts.minervaapi import MinervaApi


logger = logging.getLogger()
logger.setLevel(logging.INFO)

MINERVA_LOGIN = r'''
/*!
 * Copyright 2016 Amazon.com,
 * Inc. or its affiliates. All Rights Reserved.
 * 
 * Licensed under the Amazon Software License (the "License").
 * You may not use this file except in compliance with the
 * License. A copy of the License is located at
 * 
 *     http://aws.amazon.com/asl/
 * 
 * or in the "license" file accompanying this file. This file is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, express or implied. See the License
 * for the specific language governing permissions and
 * limitations under the License. 
 */
!function(e,t){"object"==typeof exports&&"object"==typeof module?module.exports=t():"function"==typeof define&&define.amd?define([],t):"object"==typeof exports?exports.AmazonCognitoIdentity=t():e.AmazonCognitoIdentity=t()}(this,function(){return function(e){function t(r){if(n[r])return n[r].exports;var i=n[r]={exports:{},id:r,loaded:!1};return e[r].call(i.exports,i,i.exports,t),i.loaded=!0,i.exports}var n={};return t.m=e,t.c=n,t.p="",t(0)}([function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}t.__esModule=!0;var i=n(19);Object.defineProperty(t,"AuthenticationDetails",{enumerable:!0,get:function(){return r(i).default}});var o=n(3);Object.defineProperty(t,"AuthenticationHelper",{enumerable:!0,get:function(){return r(o).default}});var s=n(5);Object.defineProperty(t,"CognitoAccessToken",{enumerable:!0,get:function(){return r(s).default}});var a=n(6);Object.defineProperty(t,"CognitoIdToken",{enumerable:!0,get:function(){return r(a).default}});var u=n(8);Object.defineProperty(t,"CognitoRefreshToken",{enumerable:!0,get:function(){return r(u).default}});var c=n(9);Object.defineProperty(t,"CognitoUser",{enumerable:!0,get:function(){return r(c).default}});var h=n(10);Object.defineProperty(t,"CognitoUserAttribute",{enumerable:!0,get:function(){return r(h).default}});var f=n(21);Object.defineProperty(t,"CognitoUserPool",{enumerable:!0,get:function(){return r(f).default}});var l=n(11);Object.defineProperty(t,"CognitoUserSession",{enumerable:!0,get:function(){return r(l).default}});var p=n(22);Object.defineProperty(t,"CookieStorage",{enumerable:!0,get:function(){return r(p).default}});var d=n(12);Object.defineProperty(t,"DateHelper",{enumerable:!0,get:function(){return r(d).default}})},function(e,t,n){(function(e){/*!
	 * The buffer module from node.js, for the browser.
	 *
	 * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
	 * @license  MIT
	 */
"use strict";function r(){try{var e=new Uint8Array(1);return e.__proto__={__proto__:Uint8Array.prototype,foo:function(){return 42}},42===e.foo()&&"function"==typeof e.subarray&&0===e.subarray(1,1).byteLength}catch(e){return!1}}function i(){return s.TYPED_ARRAY_SUPPORT?2147483647:1073741823}function o(e,t){if(i()<t)throw new RangeError("Invalid typed array length");return s.TYPED_ARRAY_SUPPORT?(e=new Uint8Array(t),e.__proto__=s.prototype):(null===e&&(e=new s(t)),e.length=t),e}function s(e,t,n){if(!(s.TYPED_ARRAY_SUPPORT||this instanceof s))return new s(e,t,n);if("number"==typeof e){if("string"==typeof t)throw new Error("If encoding is specified then the first argument must be a string");return h(this,e)}return a(this,e,t,n)}function a(e,t,n,r){if("number"==typeof t)throw new TypeError('"value" argument must not be a number');return"undefined"!=typeof ArrayBuffer&&t instanceof ArrayBuffer?p(e,t,n,r):"string"==typeof t?f(e,t,n):d(e,t)}function u(e){if("number"!=typeof e)throw new TypeError('"size" argument must be a number');if(e<0)throw new RangeError('"size" argument must not be negative')}function c(e,t,n,r){return u(t),t<=0?o(e,t):void 0!==n?"string"==typeof r?o(e,t).fill(n,r):o(e,t).fill(n):o(e,t)}function h(e,t){if(u(t),e=o(e,t<0?0:0|g(t)),!s.TYPED_ARRAY_SUPPORT)for(var n=0;n<t;++n)e[n]=0;return e}function f(e,t,n){if("string"==typeof n&&""!==n||(n="utf8"),!s.isEncoding(n))throw new TypeError('"encoding" must be a valid string encoding');var r=0|y(t,n);e=o(e,r);var i=e.write(t,n);return i!==r&&(e=e.slice(0,i)),e}function l(e,t){var n=t.length<0?0:0|g(t.length);e=o(e,n);for(var r=0;r<n;r+=1)e[r]=255&t[r];return e}function p(e,t,n,r){if(t.byteLength,n<0||t.byteLength<n)throw new RangeError("'offset' is out of bounds");if(t.byteLength<n+(r||0))throw new RangeError("'length' is out of bounds");return t=void 0===n&&void 0===r?new Uint8Array(t):void 0===r?new Uint8Array(t,n):new Uint8Array(t,n,r),s.TYPED_ARRAY_SUPPORT?(e=t,e.__proto__=s.prototype):e=l(e,t),e}function d(e,t){if(s.isBuffer(t)){var n=0|g(t.length);return e=o(e,n),0===e.length?e:(t.copy(e,0,0,n),e)}if(t){if("undefined"!=typeof ArrayBuffer&&t.buffer instanceof ArrayBuffer||"length"in t)return"number"!=typeof t.length||Z(t.length)?o(e,0):l(e,t);if("Buffer"===t.type&&$(t.data))return l(e,t.data)}throw new TypeError("First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.")}function g(e){if(e>=i())throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x"+i().toString(16)+" bytes");return 0|e}function v(e){return+e!=e&&(e=0),s.alloc(+e)}function y(e,t){if(s.isBuffer(e))return e.length;if("undefined"!=typeof ArrayBuffer&&"function"==typeof ArrayBuffer.isView&&(ArrayBuffer.isView(e)||e instanceof ArrayBuffer))return e.byteLength;"string"!=typeof e&&(e=""+e);var n=e.length;if(0===n)return 0;for(var r=!1;;)switch(t){case"ascii":case"latin1":case"binary":return n;case"utf8":case"utf-8":case void 0:return H(e).length;case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return 2*n;case"hex":return n>>>1;case"base64":return G(e).length;default:if(r)return H(e).length;t=(""+t).toLowerCase(),r=!0}}function m(e,t,n){var r=!1;if((void 0===t||t<0)&&(t=0),t>this.length)return"";if((void 0===n||n>this.length)&&(n=this.length),n<=0)return"";if(n>>>=0,t>>>=0,n<=t)return"";for(e||(e="utf8");;)switch(e){case"hex":return F(this,t,n);case"utf8":case"utf-8":return P(this,t,n);case"ascii":return b(this,t,n);case"latin1":case"binary":return k(this,t,n);case"base64":return R(this,t,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return B(this,t,n);default:if(r)throw new TypeError("Unknown encoding: "+e);e=(e+"").toLowerCase(),r=!0}}function S(e,t,n){var r=e[t];e[t]=e[n],e[n]=r}function w(e,t,n,r,i){if(0===e.length)return-1;if("string"==typeof n?(r=n,n=0):n>2147483647?n=2147483647:n<-2147483648&&(n=-2147483648),n=+n,isNaN(n)&&(n=i?0:e.length-1),n<0&&(n=e.length+n),n>=e.length){if(i)return-1;n=e.length-1}else if(n<0){if(!i)return-1;n=0}if("string"==typeof t&&(t=s.from(t,r)),s.isBuffer(t))return 0===t.length?-1:A(e,t,n,r,i);if("number"==typeof t)return t&=255,s.TYPED_ARRAY_SUPPORT&&"function"==typeof Uint8Array.prototype.indexOf?i?Uint8Array.prototype.indexOf.call(e,t,n):Uint8Array.prototype.lastIndexOf.call(e,t,n):A(e,[t],n,r,i);throw new TypeError("val must be string, number or Buffer")}function A(e,t,n,r,i){function o(e,t){return 1===s?e[t]:e.readUInt16BE(t*s)}var s=1,a=e.length,u=t.length;if(void 0!==r&&(r=String(r).toLowerCase(),"ucs2"===r||"ucs-2"===r||"utf16le"===r||"utf-16le"===r)){if(e.length<2||t.length<2)return-1;s=2,a/=2,u/=2,n/=2}var c;if(i){var h=-1;for(c=n;c<a;c++)if(o(e,c)===o(t,h===-1?0:c-h)){if(h===-1&&(h=c),c-h+1===u)return h*s}else h!==-1&&(c-=c-h),h=-1}else for(n+u>a&&(n=a-u),c=n;c>=0;c--){for(var f=!0,l=0;l<u;l++)if(o(e,c+l)!==o(t,l)){f=!1;break}if(f)return c}return-1}function C(e,t,n,r){n=Number(n)||0;var i=e.length-n;r?(r=Number(r),r>i&&(r=i)):r=i;var o=t.length;if(o%2!==0)throw new TypeError("Invalid hex string");r>o/2&&(r=o/2);for(var s=0;s<r;++s){var a=parseInt(t.substr(2*s,2),16);if(isNaN(a))return s;e[n+s]=a}return s}function U(e,t,n,r){return z(H(t,e.length-n),e,n,r)}function E(e,t,n,r){return z(J(t),e,n,r)}function T(e,t,n,r){return E(e,t,n,r)}function D(e,t,n,r){return z(G(t),e,n,r)}function I(e,t,n,r){return z(W(t,e.length-n),e,n,r)}function R(e,t,n){return 0===t&&n===e.length?X.fromByteArray(e):X.fromByteArray(e.slice(t,n))}function P(e,t,n){n=Math.min(e.length,n);for(var r=[],i=t;i<n;){var o=e[i],s=null,a=o>239?4:o>223?3:o>191?2:1;if(i+a<=n){var u,c,h,f;switch(a){case 1:o<128&&(s=o);break;case 2:u=e[i+1],128===(192&u)&&(f=(31&o)<<6|63&u,f>127&&(s=f));break;case 3:u=e[i+1],c=e[i+2],128===(192&u)&&128===(192&c)&&(f=(15&o)<<12|(63&u)<<6|63&c,f>2047&&(f<55296||f>57343)&&(s=f));break;case 4:u=e[i+1],c=e[i+2],h=e[i+3],128===(192&u)&&128===(192&c)&&128===(192&h)&&(f=(15&o)<<18|(63&u)<<12|(63&c)<<6|63&h,f>65535&&f<1114112&&(s=f))}}null===s?(s=65533,a=1):s>65535&&(s-=65536,r.push(s>>>10&1023|55296),s=56320|1023&s),r.push(s),i+=a}return _(r)}function _(e){var t=e.length;if(t<=ee)return String.fromCharCode.apply(String,e);for(var n="",r=0;r<t;)n+=String.fromCharCode.apply(String,e.slice(r,r+=ee));return n}function b(e,t,n){var r="";n=Math.min(e.length,n);for(var i=t;i<n;++i)r+=String.fromCharCode(127&e[i]);return r}function k(e,t,n){var r="";n=Math.min(e.length,n);for(var i=t;i<n;++i)r+=String.fromCharCode(e[i]);return r}function F(e,t,n){var r=e.length;(!t||t<0)&&(t=0),(!n||n<0||n>r)&&(n=r);for(var i="",o=t;o<n;++o)i+=j(e[o]);return i}function B(e,t,n){for(var r=e.slice(t,n),i="",o=0;o<r.length;o+=2)i+=String.fromCharCode(r[o]+256*r[o+1]);return i}function M(e,t,n){if(e%1!==0||e<0)throw new RangeError("offset is not uint");if(e+t>n)throw new RangeError("Trying to access beyond buffer length")}function x(e,t,n,r,i,o){if(!s.isBuffer(e))throw new TypeError('"buffer" argument must be a Buffer instance');if(t>i||t<o)throw new RangeError('"value" argument is out of bounds');if(n+r>e.length)throw new RangeError("Index out of range")}function O(e,t,n,r){t<0&&(t=65535+t+1);for(var i=0,o=Math.min(e.length-n,2);i<o;++i)e[n+i]=(t&255<<8*(r?i:1-i))>>>8*(r?i:1-i)}function N(e,t,n,r){t<0&&(t=4294967295+t+1);for(var i=0,o=Math.min(e.length-n,4);i<o;++i)e[n+i]=t>>>8*(r?i:3-i)&255}function V(e,t,n,r,i,o){if(n+r>e.length)throw new RangeError("Index out of range");if(n<0)throw new RangeError("Index out of range")}function K(e,t,n,r,i){return i||V(e,t,n,4,3.4028234663852886e38,-3.4028234663852886e38),Q.write(e,t,n,r,23,4),n+4}function q(e,t,n,r,i){return i||V(e,t,n,8,1.7976931348623157e308,-1.7976931348623157e308),Q.write(e,t,n,r,52,8),n+8}function L(e){if(e=Y(e).replace(te,""),e.length<2)return"";for(;e.length%4!==0;)e+="=";return e}function Y(e){return e.trim?e.trim():e.replace(/^\s+|\s+$/g,"")}function j(e){return e<16?"0"+e.toString(16):e.toString(16)}function H(e,t){t=t||1/0;for(var n,r=e.length,i=null,o=[],s=0;s<r;++s){if(n=e.charCodeAt(s),n>55295&&n<57344){if(!i){if(n>56319){(t-=3)>-1&&o.push(239,191,189);continue}if(s+1===r){(t-=3)>-1&&o.push(239,191,189);continue}i=n;continue}if(n<56320){(t-=3)>-1&&o.push(239,191,189),i=n;continue}n=(i-55296<<10|n-56320)+65536}else i&&(t-=3)>-1&&o.push(239,191,189);if(i=null,n<128){if((t-=1)<0)break;o.push(n)}else if(n<2048){if((t-=2)<0)break;o.push(n>>6|192,63&n|128)}else if(n<65536){if((t-=3)<0)break;o.push(n>>12|224,n>>6&63|128,63&n|128)}else{if(!(n<1114112))throw new Error("Invalid code point");if((t-=4)<0)break;o.push(n>>18|240,n>>12&63|128,n>>6&63|128,63&n|128)}}return o}function J(e){for(var t=[],n=0;n<e.length;++n)t.push(255&e.charCodeAt(n));return t}function W(e,t){for(var n,r,i,o=[],s=0;s<e.length&&!((t-=2)<0);++s)n=e.charCodeAt(s),r=n>>8,i=n%256,o.push(i),o.push(r);return o}function G(e){return X.toByteArray(L(e))}function z(e,t,n,r){for(var i=0;i<r&&!(i+n>=t.length||i>=e.length);++i)t[i+n]=e[i];return i}function Z(e){return e!==e}var X=n(15),Q=n(16),$=n(17);t.Buffer=s,t.SlowBuffer=v,t.INSPECT_MAX_BYTES=50,s.TYPED_ARRAY_SUPPORT=void 0!==e.TYPED_ARRAY_SUPPORT?e.TYPED_ARRAY_SUPPORT:r(),t.kMaxLength=i(),s.poolSize=8192,s._augment=function(e){return e.__proto__=s.prototype,e},s.from=function(e,t,n){return a(null,e,t,n)},s.TYPED_ARRAY_SUPPORT&&(s.prototype.__proto__=Uint8Array.prototype,s.__proto__=Uint8Array,"undefined"!=typeof Symbol&&Symbol.species&&s[Symbol.species]===s&&Object.defineProperty(s,Symbol.species,{value:null,configurable:!0})),s.alloc=function(e,t,n){return c(null,e,t,n)},s.allocUnsafe=function(e){return h(null,e)},s.allocUnsafeSlow=function(e){return h(null,e)},s.isBuffer=function(e){return!(null==e||!e._isBuffer)},s.compare=function(e,t){if(!s.isBuffer(e)||!s.isBuffer(t))throw new TypeError("Arguments must be Buffers");if(e===t)return 0;for(var n=e.length,r=t.length,i=0,o=Math.min(n,r);i<o;++i)if(e[i]!==t[i]){n=e[i],r=t[i];break}return n<r?-1:r<n?1:0},s.isEncoding=function(e){switch(String(e).toLowerCase()){case"hex":case"utf8":case"utf-8":case"ascii":case"latin1":case"binary":case"base64":case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return!0;default:return!1}},s.concat=function(e,t){if(!$(e))throw new TypeError('"list" argument must be an Array of Buffers');if(0===e.length)return s.alloc(0);var n;if(void 0===t)for(t=0,n=0;n<e.length;++n)t+=e[n].length;var r=s.allocUnsafe(t),i=0;for(n=0;n<e.length;++n){var o=e[n];if(!s.isBuffer(o))throw new TypeError('"list" argument must be an Array of Buffers');o.copy(r,i),i+=o.length}return r},s.byteLength=y,s.prototype._isBuffer=!0,s.prototype.swap16=function(){var e=this.length;if(e%2!==0)throw new RangeError("Buffer size must be a multiple of 16-bits");for(var t=0;t<e;t+=2)S(this,t,t+1);return this},s.prototype.swap32=function(){var e=this.length;if(e%4!==0)throw new RangeError("Buffer size must be a multiple of 32-bits");for(var t=0;t<e;t+=4)S(this,t,t+3),S(this,t+1,t+2);return this},s.prototype.swap64=function(){var e=this.length;if(e%8!==0)throw new RangeError("Buffer size must be a multiple of 64-bits");for(var t=0;t<e;t+=8)S(this,t,t+7),S(this,t+1,t+6),S(this,t+2,t+5),S(this,t+3,t+4);return this},s.prototype.toString=function(){var e=0|this.length;return 0===e?"":0===arguments.length?P(this,0,e):m.apply(this,arguments)},s.prototype.equals=function(e){if(!s.isBuffer(e))throw new TypeError("Argument must be a Buffer");return this===e||0===s.compare(this,e)},s.prototype.inspect=function(){var e="",n=t.INSPECT_MAX_BYTES;return this.length>0&&(e=this.toString("hex",0,n).match(/.{2}/g).join(" "),this.length>n&&(e+=" ... ")),"<Buffer "+e+">"},s.prototype.compare=function(e,t,n,r,i){if(!s.isBuffer(e))throw new TypeError("Argument must be a Buffer");if(void 0===t&&(t=0),void 0===n&&(n=e?e.length:0),void 0===r&&(r=0),void 0===i&&(i=this.length),t<0||n>e.length||r<0||i>this.length)throw new RangeError("out of range index");if(r>=i&&t>=n)return 0;if(r>=i)return-1;if(t>=n)return 1;if(t>>>=0,n>>>=0,r>>>=0,i>>>=0,this===e)return 0;for(var o=i-r,a=n-t,u=Math.min(o,a),c=this.slice(r,i),h=e.slice(t,n),f=0;f<u;++f)if(c[f]!==h[f]){o=c[f],a=h[f];break}return o<a?-1:a<o?1:0},s.prototype.includes=function(e,t,n){return this.indexOf(e,t,n)!==-1},s.prototype.indexOf=function(e,t,n){return w(this,e,t,n,!0)},s.prototype.lastIndexOf=function(e,t,n){return w(this,e,t,n,!1)},s.prototype.write=function(e,t,n,r){if(void 0===t)r="utf8",n=this.length,t=0;else if(void 0===n&&"string"==typeof t)r=t,n=this.length,t=0;else{if(!isFinite(t))throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");t|=0,isFinite(n)?(n|=0,void 0===r&&(r="utf8")):(r=n,n=void 0)}var i=this.length-t;if((void 0===n||n>i)&&(n=i),e.length>0&&(n<0||t<0)||t>this.length)throw new RangeError("Attempt to write outside buffer bounds");r||(r="utf8");for(var o=!1;;)switch(r){case"hex":return C(this,e,t,n);case"utf8":case"utf-8":return U(this,e,t,n);case"ascii":return E(this,e,t,n);case"latin1":case"binary":return T(this,e,t,n);case"base64":return D(this,e,t,n);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return I(this,e,t,n);default:if(o)throw new TypeError("Unknown encoding: "+r);r=(""+r).toLowerCase(),o=!0}},s.prototype.toJSON=function(){return{type:"Buffer",data:Array.prototype.slice.call(this._arr||this,0)}};var ee=4096;s.prototype.slice=function(e,t){var n=this.length;e=~~e,t=void 0===t?n:~~t,e<0?(e+=n,e<0&&(e=0)):e>n&&(e=n),t<0?(t+=n,t<0&&(t=0)):t>n&&(t=n),t<e&&(t=e);var r;if(s.TYPED_ARRAY_SUPPORT)r=this.subarray(e,t),r.__proto__=s.prototype;else{var i=t-e;r=new s(i,void 0);for(var o=0;o<i;++o)r[o]=this[o+e]}return r},s.prototype.readUIntLE=function(e,t,n){e|=0,t|=0,n||M(e,t,this.length);for(var r=this[e],i=1,o=0;++o<t&&(i*=256);)r+=this[e+o]*i;return r},s.prototype.readUIntBE=function(e,t,n){e|=0,t|=0,n||M(e,t,this.length);for(var r=this[e+--t],i=1;t>0&&(i*=256);)r+=this[e+--t]*i;return r},s.prototype.readUInt8=function(e,t){return t||M(e,1,this.length),this[e]},s.prototype.readUInt16LE=function(e,t){return t||M(e,2,this.length),this[e]|this[e+1]<<8},s.prototype.readUInt16BE=function(e,t){return t||M(e,2,this.length),this[e]<<8|this[e+1]},s.prototype.readUInt32LE=function(e,t){return t||M(e,4,this.length),(this[e]|this[e+1]<<8|this[e+2]<<16)+16777216*this[e+3]},s.prototype.readUInt32BE=function(e,t){return t||M(e,4,this.length),16777216*this[e]+(this[e+1]<<16|this[e+2]<<8|this[e+3])},s.prototype.readIntLE=function(e,t,n){e|=0,t|=0,n||M(e,t,this.length);for(var r=this[e],i=1,o=0;++o<t&&(i*=256);)r+=this[e+o]*i;return i*=128,r>=i&&(r-=Math.pow(2,8*t)),r},s.prototype.readIntBE=function(e,t,n){e|=0,t|=0,n||M(e,t,this.length);for(var r=t,i=1,o=this[e+--r];r>0&&(i*=256);)o+=this[e+--r]*i;return i*=128,o>=i&&(o-=Math.pow(2,8*t)),o},s.prototype.readInt8=function(e,t){return t||M(e,1,this.length),128&this[e]?(255-this[e]+1)*-1:this[e]},s.prototype.readInt16LE=function(e,t){t||M(e,2,this.length);var n=this[e]|this[e+1]<<8;return 32768&n?4294901760|n:n},s.prototype.readInt16BE=function(e,t){t||M(e,2,this.length);var n=this[e+1]|this[e]<<8;return 32768&n?4294901760|n:n},s.prototype.readInt32LE=function(e,t){return t||M(e,4,this.length),this[e]|this[e+1]<<8|this[e+2]<<16|this[e+3]<<24},s.prototype.readInt32BE=function(e,t){return t||M(e,4,this.length),this[e]<<24|this[e+1]<<16|this[e+2]<<8|this[e+3]},s.prototype.readFloatLE=function(e,t){return t||M(e,4,this.length),Q.read(this,e,!0,23,4)},s.prototype.readFloatBE=function(e,t){return t||M(e,4,this.length),Q.read(this,e,!1,23,4)},s.prototype.readDoubleLE=function(e,t){return t||M(e,8,this.length),Q.read(this,e,!0,52,8)},s.prototype.readDoubleBE=function(e,t){return t||M(e,8,this.length),Q.read(this,e,!1,52,8)},s.prototype.writeUIntLE=function(e,t,n,r){if(e=+e,t|=0,n|=0,!r){var i=Math.pow(2,8*n)-1;x(this,e,t,n,i,0)}var o=1,s=0;for(this[t]=255&e;++s<n&&(o*=256);)this[t+s]=e/o&255;return t+n},s.prototype.writeUIntBE=function(e,t,n,r){if(e=+e,t|=0,n|=0,!r){var i=Math.pow(2,8*n)-1;x(this,e,t,n,i,0)}var o=n-1,s=1;for(this[t+o]=255&e;--o>=0&&(s*=256);)this[t+o]=e/s&255;return t+n},s.prototype.writeUInt8=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,1,255,0),s.TYPED_ARRAY_SUPPORT||(e=Math.floor(e)),this[t]=255&e,t+1},s.prototype.writeUInt16LE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,2,65535,0),s.TYPED_ARRAY_SUPPORT?(this[t]=255&e,this[t+1]=e>>>8):O(this,e,t,!0),t+2},s.prototype.writeUInt16BE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,2,65535,0),s.TYPED_ARRAY_SUPPORT?(this[t]=e>>>8,this[t+1]=255&e):O(this,e,t,!1),t+2},s.prototype.writeUInt32LE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,4,4294967295,0),s.TYPED_ARRAY_SUPPORT?(this[t+3]=e>>>24,this[t+2]=e>>>16,this[t+1]=e>>>8,this[t]=255&e):N(this,e,t,!0),t+4},s.prototype.writeUInt32BE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,4,4294967295,0),s.TYPED_ARRAY_SUPPORT?(this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=255&e):N(this,e,t,!1),t+4},s.prototype.writeIntLE=function(e,t,n,r){if(e=+e,t|=0,!r){var i=Math.pow(2,8*n-1);x(this,e,t,n,i-1,-i)}var o=0,s=1,a=0;for(this[t]=255&e;++o<n&&(s*=256);)e<0&&0===a&&0!==this[t+o-1]&&(a=1),this[t+o]=(e/s>>0)-a&255;return t+n},s.prototype.writeIntBE=function(e,t,n,r){if(e=+e,t|=0,!r){var i=Math.pow(2,8*n-1);x(this,e,t,n,i-1,-i)}var o=n-1,s=1,a=0;for(this[t+o]=255&e;--o>=0&&(s*=256);)e<0&&0===a&&0!==this[t+o+1]&&(a=1),this[t+o]=(e/s>>0)-a&255;return t+n},s.prototype.writeInt8=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,1,127,-128),s.TYPED_ARRAY_SUPPORT||(e=Math.floor(e)),e<0&&(e=255+e+1),this[t]=255&e,t+1},s.prototype.writeInt16LE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,2,32767,-32768),s.TYPED_ARRAY_SUPPORT?(this[t]=255&e,this[t+1]=e>>>8):O(this,e,t,!0),t+2},s.prototype.writeInt16BE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,2,32767,-32768),s.TYPED_ARRAY_SUPPORT?(this[t]=e>>>8,this[t+1]=255&e):O(this,e,t,!1),t+2},s.prototype.writeInt32LE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,4,2147483647,-2147483648),s.TYPED_ARRAY_SUPPORT?(this[t]=255&e,this[t+1]=e>>>8,this[t+2]=e>>>16,this[t+3]=e>>>24):N(this,e,t,!0),t+4},s.prototype.writeInt32BE=function(e,t,n){return e=+e,t|=0,n||x(this,e,t,4,2147483647,-2147483648),e<0&&(e=4294967295+e+1),s.TYPED_ARRAY_SUPPORT?(this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=255&e):N(this,e,t,!1),t+4},s.prototype.writeFloatLE=function(e,t,n){return K(this,e,t,!0,n)},s.prototype.writeFloatBE=function(e,t,n){return K(this,e,t,!1,n)},s.prototype.writeDoubleLE=function(e,t,n){return q(this,e,t,!0,n)},s.prototype.writeDoubleBE=function(e,t,n){return q(this,e,t,!1,n)},s.prototype.copy=function(e,t,n,r){if(n||(n=0),r||0===r||(r=this.length),t>=e.length&&(t=e.length),t||(t=0),r>0&&r<n&&(r=n),r===n)return 0;if(0===e.length||0===this.length)return 0;if(t<0)throw new RangeError("targetStart out of bounds");if(n<0||n>=this.length)throw new RangeError("sourceStart out of bounds");if(r<0)throw new RangeError("sourceEnd out of bounds");r>this.length&&(r=this.length),e.length-t<r-n&&(r=e.length-t+n);var i,o=r-n;if(this===e&&n<t&&t<r)for(i=o-1;i>=0;--i)e[i+t]=this[i+n];else if(o<1e3||!s.TYPED_ARRAY_SUPPORT)for(i=0;i<o;++i)e[i+t]=this[i+n];else Uint8Array.prototype.set.call(e,this.subarray(n,n+o),t);return o},s.prototype.fill=function(e,t,n,r){if("string"==typeof e){if("string"==typeof t?(r=t,t=0,n=this.length):"string"==typeof n&&(r=n,n=this.length),1===e.length){var i=e.charCodeAt(0);i<256&&(e=i)}if(void 0!==r&&"string"!=typeof r)throw new TypeError("encoding must be a string");if("string"==typeof r&&!s.isEncoding(r))throw new TypeError("Unknown encoding: "+r)}else"number"==typeof e&&(e&=255);if(t<0||this.length<t||this.length<n)throw new RangeError("Out of range index");if(n<=t)return this;t>>>=0,n=void 0===n?this.length:n>>>0,e||(e=0);var o;if("number"==typeof e)for(o=t;o<n;++o)this[o]=e;else{var a=s.isBuffer(e)?e:H(new s(e,r).toString()),u=a.length;for(o=0;o<n-t;++o)this[o+t]=a[o%u]}return this};var te=/[^+\/0-9A-Za-z-_]/g}).call(t,function(){return this}())},function(e,t,n){function r(e,t){if(e.length%a!==0){var n=e.length+(a-e.length%a);e=s.concat([e,u],n)}for(var r=[],i=t?e.readInt32BE:e.readInt32LE,o=0;o<e.length;o+=a)r.push(i.call(e,o));return r}function i(e,t,n){for(var r=new s(t),i=n?r.writeInt32BE:r.writeInt32LE,o=0;o<e.length;o++)i.call(r,e[o],4*o,!0);return r}function o(e,t,n,o){s.isBuffer(e)||(e=new s(e));var a=t(r(e,o),e.length*c);return i(a,n,o)}var s=n(1).Buffer,a=4,u=new s(a);u.fill(0);var c=8;e.exports={hash:o}},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&(t[n]=e[n]);return t.default=e,t}function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var s=n(1),a=n(14),u=i(a),c=n(4),h=r(c),f=u.createHash,l=u.createHmac,p=u.randomBytes,d="FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF",g="userAttributes.",v=function(){function e(t){o(this,e),this.N=new h.default(d,16),this.g=new h.default("2",16),this.k=new h.default(this.hexHash("00"+this.N.toString(16)+"0"+this.g.toString(16)),16),this.smallAValue=this.generateRandomSmallA(),this.getLargeAValue(function(){}),this.infoBits=s.Buffer.from("Caldera Derived Key","utf8"),this.poolName=t}return e.prototype.getSmallAValue=function(){return this.smallAValue},e.prototype.getLargeAValue=function(e){var t=this;this.largeAValue?e(null,this.largeAValue):this.calculateA(this.smallAValue,function(n,r){n&&e(n,null),t.largeAValue=r,e(null,t.largeAValue)})},e.prototype.generateRandomSmallA=function(){var e=p(128).toString("hex"),t=new h.default(e,16),n=t.mod(this.N);return n},e.prototype.generateRandomString=function(){return p(40).toString("base64")},e.prototype.getRandomPassword=function(){return this.randomPassword},e.prototype.getSaltDevices=function(){return this.SaltToHashDevices},e.prototype.getVerifierDevices=function(){return this.verifierDevices},e.prototype.generateHashDevice=function(e,t,n){var r=this;this.randomPassword=this.generateRandomString();var i=""+e+t+":"+this.randomPassword,o=this.hash(i),s=p(16).toString("hex");this.SaltToHashDevices=this.padHex(new h.default(s,16)),this.g.modPow(new h.default(this.hexHash(this.SaltToHashDevices+o),16),this.N,function(e,t){e&&n(e,null),r.verifierDevices=r.padHex(t),n(null,null)})},e.prototype.calculateA=function(e,t){var n=this;this.g.modPow(e,this.N,function(e,r){e&&t(e,null),r.mod(n.N).equals(h.default.ZERO)&&t(new Error("Illegal paramater. A mod N cannot be 0."),null),t(null,r)})},e.prototype.calculateU=function(e,t){this.UHexHash=this.hexHash(this.padHex(e)+this.padHex(t));var n=new h.default(this.UHexHash,16);return n},e.prototype.hash=function(e){var t=f("sha256").update(e).digest("hex");return new Array(64-t.length).join("0")+t},e.prototype.hexHash=function(e){return this.hash(s.Buffer.from(e,"hex"))},e.prototype.computehkdf=function(e,t){var n=l("sha256",t).update(e).digest(),r=s.Buffer.concat([this.infoBits,s.Buffer.from(String.fromCharCode(1),"utf8")]),i=l("sha256",n).update(r).digest();return i.slice(0,16)},e.prototype.getPasswordAuthenticationKey=function(e,t,n,r,i){var o=this;if(n.mod(this.N).equals(h.default.ZERO))throw new Error("B cannot be zero.");if(this.UValue=this.calculateU(this.largeAValue,n),this.UValue.equals(h.default.ZERO))throw new Error("U cannot be zero.");var a=""+this.poolName+e+":"+t,u=this.hash(a),c=new h.default(this.hexHash(this.padHex(r)+u),16);this.calculateS(c,n,function(e,t){e&&i(e,null);var n=o.computehkdf(s.Buffer.from(o.padHex(t),"hex"),s.Buffer.from(o.padHex(o.UValue.toString(16)),"hex"));i(null,n)})},e.prototype.calculateS=function(e,t,n){var r=this;this.g.modPow(e,this.N,function(i,o){i&&n(i,null);var s=t.subtract(r.k.multiply(o));s.modPow(r.smallAValue.add(r.UValue.multiply(e)),r.N,function(e,t){e&&n(e,null),n(null,t.mod(r.N))})})},e.prototype.getNewPasswordRequiredChallengeUserAttributePrefix=function(){return g},e.prototype.padHex=function(e){var t=e.toString(16);return t.length%2===1?t="0"+t:"89ABCDEFabcdef".indexOf(t[0])!==-1&&(t="00"+t),t},e}();t.default=v},function(e,t){"use strict";function n(e,t){null!=e&&this.fromString(e,t)}function r(){return new n(null)}function i(e,t,n,r,i,o){for(;--o>=0;){var s=t*this[e++]+n[r]+i;i=Math.floor(s/67108864),n[r++]=67108863&s}return i}function o(e,t,n,r,i,o){for(var s=32767&t,a=t>>15;--o>=0;){var u=32767&this[e],c=this[e++]>>15,h=a*u+c*s;u=s*u+((32767&h)<<15)+n[r]+(1073741823&i),i=(u>>>30)+(h>>>15)+a*c+(i>>>30),n[r++]=1073741823&u}return i}function s(e,t,n,r,i,o){for(var s=16383&t,a=t>>14;--o>=0;){var u=16383&this[e],c=this[e++]>>14,h=a*u+c*s;u=s*u+((16383&h)<<14)+n[r]+i,i=(u>>28)+(h>>14)+a*c,n[r++]=268435455&u}return i}function a(e){return Z.charAt(e)}function u(e,t){var n=X[e.charCodeAt(t)];return null==n?-1:n}function c(e){for(var t=this.t-1;t>=0;--t)e[t]=this[t];e.t=this.t,e.s=this.s}function h(e){this.t=1,this.s=e<0?-1:0,e>0?this[0]=e:e<-1?this[0]=e+this.DV:this.t=0}function f(e){var t=r();return t.fromInt(e),t}function l(e,t){var r;if(16==t)r=4;else if(8==t)r=3;else if(2==t)r=1;else if(32==t)r=5;else{if(4!=t)throw new Error("Only radix 2, 4, 8, 16, 32 are supported");r=2}this.t=0,this.s=0;for(var i=e.length,o=!1,s=0;--i>=0;){var a=u(e,i);a<0?"-"==e.charAt(i)&&(o=!0):(o=!1,0==s?this[this.t++]=a:s+r>this.DB?(this[this.t-1]|=(a&(1<<this.DB-s)-1)<<s,this[this.t++]=a>>this.DB-s):this[this.t-1]|=a<<s,s+=r,s>=this.DB&&(s-=this.DB))}this.clamp(),o&&n.ZERO.subTo(this,this)}function p(){for(var e=this.s&this.DM;this.t>0&&this[this.t-1]==e;)--this.t}function d(e){if(this.s<0)return"-"+this.negate().toString();var t;if(16==e)t=4;else if(8==e)t=3;else if(2==e)t=1;else if(32==e)t=5;else{if(4!=e)throw new Error("Only radix 2, 4, 8, 16, 32 are supported");t=2}var n,r=(1<<t)-1,i=!1,o="",s=this.t,u=this.DB-s*this.DB%t;if(s-- >0)for(u<this.DB&&(n=this[s]>>u)>0&&(i=!0,o=a(n));s>=0;)u<t?(n=(this[s]&(1<<u)-1)<<t-u,n|=this[--s]>>(u+=this.DB-t)):(n=this[s]>>(u-=t)&r,u<=0&&(u+=this.DB,--s)),n>0&&(i=!0),i&&(o+=a(n));return i?o:"0"}function g(){var e=r();return n.ZERO.subTo(this,e),e}function v(){return this.s<0?this.negate():this}function y(e){var t=this.s-e.s;if(0!=t)return t;var n=this.t;if(t=n-e.t,0!=t)return this.s<0?-t:t;for(;--n>=0;)if(0!=(t=this[n]-e[n]))return t;return 0}function m(e){var t,n=1;return 0!=(t=e>>>16)&&(e=t,n+=16),0!=(t=e>>8)&&(e=t,n+=8),0!=(t=e>>4)&&(e=t,n+=4),0!=(t=e>>2)&&(e=t,n+=2),0!=(t=e>>1)&&(e=t,n+=1),n}function S(){return this.t<=0?0:this.DB*(this.t-1)+m(this[this.t-1]^this.s&this.DM)}function w(e,t){var n;for(n=this.t-1;n>=0;--n)t[n+e]=this[n];for(n=e-1;n>=0;--n)t[n]=0;t.t=this.t+e,t.s=this.s}function A(e,t){for(var n=e;n<this.t;++n)t[n-e]=this[n];t.t=Math.max(this.t-e,0),t.s=this.s}function C(e,t){var n,r=e%this.DB,i=this.DB-r,o=(1<<i)-1,s=Math.floor(e/this.DB),a=this.s<<r&this.DM;for(n=this.t-1;n>=0;--n)t[n+s+1]=this[n]>>i|a,a=(this[n]&o)<<r;for(n=s-1;n>=0;--n)t[n]=0;t[s]=a,t.t=this.t+s+1,t.s=this.s,t.clamp()}function U(e,t){t.s=this.s;var n=Math.floor(e/this.DB);if(n>=this.t)return void(t.t=0);var r=e%this.DB,i=this.DB-r,o=(1<<r)-1;t[0]=this[n]>>r;for(var s=n+1;s<this.t;++s)t[s-n-1]|=(this[s]&o)<<i,t[s-n]=this[s]>>r;r>0&&(t[this.t-n-1]|=(this.s&o)<<i),t.t=this.t-n,t.clamp()}function E(e,t){for(var n=0,r=0,i=Math.min(e.t,this.t);n<i;)r+=this[n]-e[n],t[n++]=r&this.DM,r>>=this.DB;if(e.t<this.t){for(r-=e.s;n<this.t;)r+=this[n],t[n++]=r&this.DM,r>>=this.DB;r+=this.s}else{for(r+=this.s;n<e.t;)r-=e[n],t[n++]=r&this.DM,r>>=this.DB;r-=e.s}t.s=r<0?-1:0,r<-1?t[n++]=this.DV+r:r>0&&(t[n++]=r),t.t=n,t.clamp()}function T(e,t){var r=this.abs(),i=e.abs(),o=r.t;for(t.t=o+i.t;--o>=0;)t[o]=0;for(o=0;o<i.t;++o)t[o+r.t]=r.am(0,i[o],t,o,0,r.t);t.s=0,t.clamp(),this.s!=e.s&&n.ZERO.subTo(t,t)}function D(e){for(var t=this.abs(),n=e.t=2*t.t;--n>=0;)e[n]=0;for(n=0;n<t.t-1;++n){var r=t.am(n,t[n],e,2*n,0,1);(e[n+t.t]+=t.am(n+1,2*t[n],e,2*n+1,r,t.t-n-1))>=t.DV&&(e[n+t.t]-=t.DV,e[n+t.t+1]=1)}e.t>0&&(e[e.t-1]+=t.am(n,t[n],e,2*n,0,1)),e.s=0,e.clamp()}function I(e,t,i){var o=e.abs();if(!(o.t<=0)){var s=this.abs();if(s.t<o.t)return null!=t&&t.fromInt(0),void(null!=i&&this.copyTo(i));null==i&&(i=r());var a=r(),u=this.s,c=e.s,h=this.DB-m(o[o.t-1]);h>0?(o.lShiftTo(h,a),s.lShiftTo(h,i)):(o.copyTo(a),s.copyTo(i));var f=a.t,l=a[f-1];if(0!=l){var p=l*(1<<this.F1)+(f>1?a[f-2]>>this.F2:0),d=this.FV/p,g=(1<<this.F1)/p,v=1<<this.F2,y=i.t,S=y-f,w=null==t?r():t;for(a.dlShiftTo(S,w),i.compareTo(w)>=0&&(i[i.t++]=1,i.subTo(w,i)),n.ONE.dlShiftTo(f,w),w.subTo(a,a);a.t<f;)a[a.t++]=0;for(;--S>=0;){var A=i[--y]==l?this.DM:Math.floor(i[y]*d+(i[y-1]+v)*g);if((i[y]+=a.am(0,A,i,S,0,f))<A)for(a.dlShiftTo(S,w),i.subTo(w,i);i[y]<--A;)i.subTo(w,i)}null!=t&&(i.drShiftTo(f,t),u!=c&&n.ZERO.subTo(t,t)),i.t=f,i.clamp(),h>0&&i.rShiftTo(h,i),u<0&&n.ZERO.subTo(i,i)}}}function R(e){var t=r();return this.abs().divRemTo(e,null,t),this.s<0&&t.compareTo(n.ZERO)>0&&e.subTo(t,t),t}function P(){if(this.t<1)return 0;var e=this[0];if(0==(1&e))return 0;var t=3&e;return t=t*(2-(15&e)*t)&15,t=t*(2-(255&e)*t)&255,t=t*(2-((65535&e)*t&65535))&65535,t=t*(2-e*t%this.DV)%this.DV,t>0?this.DV-t:-t}function _(e){return 0==this.compareTo(e)}function b(e,t){for(var n=0,r=0,i=Math.min(e.t,this.t);n<i;)r+=this[n]+e[n],t[n++]=r&this.DM,r>>=this.DB;if(e.t<this.t){for(r+=e.s;n<this.t;)r+=this[n],t[n++]=r&this.DM,r>>=this.DB;r+=this.s}else{for(r+=this.s;n<e.t;)r+=e[n],t[n++]=r&this.DM,r>>=this.DB;r+=e.s}t.s=r<0?-1:0,r>0?t[n++]=r:r<-1&&(t[n++]=this.DV+r),t.t=n,t.clamp()}function k(e){var t=r();return this.addTo(e,t),t}function F(e){var t=r();return this.subTo(e,t),t}function B(e){var t=r();return this.multiplyTo(e,t),t}function M(e){var t=r();return this.divRemTo(e,t,null),t}function x(e){this.m=e,this.mp=e.invDigit(),this.mpl=32767&this.mp,this.mph=this.mp>>15,this.um=(1<<e.DB-15)-1,this.mt2=2*e.t}function O(e){var t=r();return e.abs().dlShiftTo(this.m.t,t),t.divRemTo(this.m,null,t),e.s<0&&t.compareTo(n.ZERO)>0&&this.m.subTo(t,t),t}function N(e){var t=r();return e.copyTo(t),this.reduce(t),t}function V(e){for(;e.t<=this.mt2;)e[e.t++]=0;for(var t=0;t<this.m.t;++t){var n=32767&e[t],r=n*this.mpl+((n*this.mph+(e[t]>>15)*this.mpl&this.um)<<15)&e.DM;for(n=t+this.m.t,e[n]+=this.m.am(0,r,e,t,0,this.m.t);e[n]>=e.DV;)e[n]-=e.DV,e[++n]++}e.clamp(),e.drShiftTo(this.m.t,e),e.compareTo(this.m)>=0&&e.subTo(this.m,e)}function K(e,t){e.squareTo(t),this.reduce(t)}function q(e,t,n){e.multiplyTo(t,n),this.reduce(n)}function L(e,t,n){var i,o=e.bitLength(),s=f(1),a=new x(t);if(o<=0)return s;i=o<18?1:o<48?3:o<144?4:o<768?5:6;var u=new Array,c=3,h=i-1,l=(1<<i)-1;if(u[1]=a.convert(this),i>1){var p=r();for(a.sqrTo(u[1],p);c<=l;)u[c]=r(),a.mulTo(p,u[c-2],u[c]),c+=2}var d,g,v=e.t-1,y=!0,S=r();for(o=m(e[v])-1;v>=0;){for(o>=h?d=e[v]>>o-h&l:(d=(e[v]&(1<<o+1)-1)<<h-o,v>0&&(d|=e[v-1]>>this.DB+o-h)),c=i;0==(1&d);)d>>=1,--c;if((o-=c)<0&&(o+=this.DB,--v),y)u[d].copyTo(s),y=!1;else{for(;c>1;)a.sqrTo(s,S),a.sqrTo(S,s),c-=2;c>0?a.sqrTo(s,S):(g=s,s=S,S=g),a.mulTo(S,u[d],s)}for(;v>=0&&0==(e[v]&1<<o);)a.sqrTo(s,S),g=s,s=S,S=g,--o<0&&(o=this.DB-1,--v)}var w=a.revert(s);return n(null,w),w}t.__esModule=!0,t.default=n;var Y,j=0xdeadbeefcafe,H=15715070==(16777215&j),J="undefined"!=typeof navigator;J&&H&&"Microsoft Internet Explorer"==navigator.appName?(n.prototype.am=o,Y=30):J&&H&&"Netscape"!=navigator.appName?(n.prototype.am=i,Y=26):(n.prototype.am=s,Y=28),n.prototype.DB=Y,
n.prototype.DM=(1<<Y)-1,n.prototype.DV=1<<Y;var W=52;n.prototype.FV=Math.pow(2,W),n.prototype.F1=W-Y,n.prototype.F2=2*Y-W;var G,z,Z="0123456789abcdefghijklmnopqrstuvwxyz",X=new Array;for(G="0".charCodeAt(0),z=0;z<=9;++z)X[G++]=z;for(G="a".charCodeAt(0),z=10;z<36;++z)X[G++]=z;for(G="A".charCodeAt(0),z=10;z<36;++z)X[G++]=z;x.prototype.convert=O,x.prototype.revert=N,x.prototype.reduce=V,x.prototype.mulTo=q,x.prototype.sqrTo=K,n.prototype.copyTo=c,n.prototype.fromInt=h,n.prototype.fromString=l,n.prototype.clamp=p,n.prototype.dlShiftTo=w,n.prototype.drShiftTo=A,n.prototype.lShiftTo=C,n.prototype.rShiftTo=U,n.prototype.subTo=E,n.prototype.multiplyTo=T,n.prototype.squareTo=D,n.prototype.divRemTo=I,n.prototype.invDigit=P,n.prototype.addTo=b,n.prototype.toString=d,n.prototype.negate=g,n.prototype.abs=v,n.prototype.compareTo=y,n.prototype.bitLength=S,n.prototype.mod=R,n.prototype.equals=_,n.prototype.add=k,n.prototype.subtract=F,n.prototype.multiply=B,n.prototype.divide=M,n.prototype.modPow=L,n.ZERO=f(0),n.ONE=f(1)},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function o(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}t.__esModule=!0;var a=n(7),u=r(a),c=function(e){function t(){var n=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},r=n.AccessToken;return i(this,t),o(this,e.call(this,r||""))}return s(t,e),t}(u.default);t.default=c},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function o(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}t.__esModule=!0;var a=n(7),u=r(a),c=function(e){function t(){var n=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},r=n.IdToken;return i(this,t),o(this,e.call(this,r||""))}return s(t,e),t}(u.default);t.default=c},function(e,t,n){"use strict";function r(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var i=n(1),o=function(){function e(t){r(this,e),this.jwtToken=t||"",this.payload=this.decodePayload()}return e.prototype.getJwtToken=function(){return this.jwtToken},e.prototype.getExpiration=function(){return this.payload.exp},e.prototype.getIssuedAt=function(){return this.payload.iat},e.prototype.decodePayload=function(){var e=this.jwtToken.split(".")[1];try{return JSON.parse(i.Buffer.from(e,"base64").toString("utf8"))}catch(e){return{}}},e}();t.default=o},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r=function(){function e(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},r=t.RefreshToken;n(this,e),this.token=r||""}return e.prototype.getToken=function(){return this.token},e}();t.default=r},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&(t[n]=e[n]);return t.default=e,t}function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var s=n(1),a=n(14),u=i(a),c=n(4),h=r(c),f=n(3),l=r(f),p=n(5),d=r(p),g=n(6),v=r(g),y=n(8),m=r(y),S=n(11),w=r(S),A=n(12),C=r(A),U=n(10),E=r(U),T=n(13),D=r(T),I=u.createHmac,R=function(){function e(t){if(o(this,e),null==t||null==t.Username||null==t.Pool)throw new Error("Username and pool information are required.");this.username=t.Username||"",this.pool=t.Pool,this.Session=null,this.client=t.Pool.client,this.signInUserSession=null,this.authenticationFlowType="USER_SRP_AUTH",this.storage=t.Storage||(new D.default).getStorage()}return e.prototype.setSignInUserSession=function(e){this.clearCachedTokens(),this.signInUserSession=e,this.cacheTokens()},e.prototype.getSignInUserSession=function(){return this.signInUserSession},e.prototype.getUsername=function(){return this.username},e.prototype.getAuthenticationFlowType=function(){return this.authenticationFlowType},e.prototype.setAuthenticationFlowType=function(e){this.authenticationFlowType=e},e.prototype.initiateAuth=function(e,t){var n=this,r=e.getAuthParameters();r.USERNAME=this.username;var i={AuthFlow:"CUSTOM_AUTH",ClientId:this.pool.getClientId(),AuthParameters:r,ClientMetadata:e.getValidationData()};this.getUserContextData()&&(i.UserContextData=this.getUserContextData()),this.client.request("InitiateAuth",i,function(e,r){if(e)return t.onFailure(e);var i=r.ChallengeName,o=r.ChallengeParameters;return"CUSTOM_CHALLENGE"===i?(n.Session=r.Session,t.customChallenge(o)):(n.signInUserSession=n.getCognitoUserSession(r.AuthenticationResult),n.cacheTokens(),t.onSuccess(n.signInUserSession))})},e.prototype.authenticateUser=function(e,t){return"USER_PASSWORD_AUTH"===this.authenticationFlowType?this.authenticateUserPlainUsernamePassword(e,t):"USER_SRP_AUTH"===this.authenticationFlowType?this.authenticateUserDefaultAuth(e,t):t.onFailure(new Error("Authentication flow type is invalid."))},e.prototype.authenticateUserDefaultAuth=function(e,t){var n=this,r=new l.default(this.pool.getUserPoolId().split("_")[1]),i=new C.default,o=void 0,a=void 0,u={};null!=this.deviceKey&&(u.DEVICE_KEY=this.deviceKey),u.USERNAME=this.username,r.getLargeAValue(function(c,f){c&&t.onFailure(c),u.SRP_A=f.toString(16),"CUSTOM_AUTH"===n.authenticationFlowType&&(u.CHALLENGE_NAME="SRP_A");var l={AuthFlow:n.authenticationFlowType,ClientId:n.pool.getClientId(),AuthParameters:u,ClientMetadata:e.getValidationData()};n.getUserContextData(n.username)&&(l.UserContextData=n.getUserContextData(n.username)),n.client.request("InitiateAuth",l,function(u,c){if(u)return t.onFailure(u);var f=c.ChallengeParameters;n.username=f.USER_ID_FOR_SRP,o=new h.default(f.SRP_B,16),a=new h.default(f.SALT,16),n.getCachedDeviceKeyAndPassword(),r.getPasswordAuthenticationKey(n.username,e.getPassword(),o,a,function(e,o){e&&t.onFailure(e);var a=i.getNowString(),u=I("sha256",o).update(s.Buffer.concat([s.Buffer.from(n.pool.getUserPoolId().split("_")[1],"utf8"),s.Buffer.from(n.username,"utf8"),s.Buffer.from(f.SECRET_BLOCK,"base64"),s.Buffer.from(a,"utf8")])).digest("base64"),h={};h.USERNAME=n.username,h.PASSWORD_CLAIM_SECRET_BLOCK=f.SECRET_BLOCK,h.TIMESTAMP=a,h.PASSWORD_CLAIM_SIGNATURE=u,null!=n.deviceKey&&(h.DEVICE_KEY=n.deviceKey);var l=function e(t,r){return n.client.request("RespondToAuthChallenge",t,function(i,o){return i&&"ResourceNotFoundException"===i.code&&i.message.toLowerCase().indexOf("device")!==-1?(h.DEVICE_KEY=null,n.deviceKey=null,n.randomPassword=null,n.deviceGroupKey=null,n.clearCachedDeviceKeyAndPassword(),e(t,r)):r(i,o)})},p={ChallengeName:"PASSWORD_VERIFIER",ClientId:n.pool.getClientId(),ChallengeResponses:h,Session:c.Session};n.getUserContextData()&&(p.UserContextData=n.getUserContextData()),l(p,function(e,i){if(e)return t.onFailure(e);var o=i.ChallengeName;if("NEW_PASSWORD_REQUIRED"===o){n.Session=i.Session;var s=null,a=null,u=[],c=r.getNewPasswordRequiredChallengeUserAttributePrefix();if(i.ChallengeParameters&&(s=JSON.parse(i.ChallengeParameters.userAttributes),a=JSON.parse(i.ChallengeParameters.requiredAttributes)),a)for(var h=0;h<a.length;h++)u[h]=a[h].substr(c.length);return t.newPasswordRequired(s,u)}return n.authenticateUserInternal(i,r,t)})})})})},e.prototype.authenticateUserPlainUsernamePassword=function(e,t){var n=this,r={};if(r.USERNAME=this.username,r.PASSWORD=e.getPassword(),!r.PASSWORD)return void t.onFailure(new Error("PASSWORD parameter is required"));var i=new l.default(this.pool.getUserPoolId().split("_")[1]);this.getCachedDeviceKeyAndPassword(),null!=this.deviceKey&&(r.DEVICE_KEY=this.deviceKey);var o={AuthFlow:"USER_PASSWORD_AUTH",ClientId:this.pool.getClientId(),AuthParameters:r,ClientMetadata:e.getValidationData()};this.getUserContextData(this.username)&&(o.UserContextData=this.getUserContextData(this.username)),this.client.request("InitiateAuth",o,function(e,r){return e?t.onFailure(e):n.authenticateUserInternal(r,i,t)})},e.prototype.authenticateUserInternal=function(e,t,n){var r=this,i=e.ChallengeName,o=e.ChallengeParameters;if("SMS_MFA"===i)return this.Session=e.Session,n.mfaRequired(i,o);if("SELECT_MFA_TYPE"===i)return this.Session=e.Session,n.selectMFAType(i,o);if("MFA_SETUP"===i)return this.Session=e.Session,n.mfaSetup(i,o);if("SOFTWARE_TOKEN_MFA"===i)return this.Session=e.Session,n.totpRequired(i,o);if("CUSTOM_CHALLENGE"===i)return this.Session=e.Session,n.customChallenge(o);if("DEVICE_SRP_AUTH"===i)return void this.getDeviceResponse(n);this.signInUserSession=this.getCognitoUserSession(e.AuthenticationResult),this.cacheTokens();var a=e.AuthenticationResult.NewDeviceMetadata;return null==a?n.onSuccess(this.signInUserSession):void t.generateHashDevice(e.AuthenticationResult.NewDeviceMetadata.DeviceGroupKey,e.AuthenticationResult.NewDeviceMetadata.DeviceKey,function(i){if(i)return n.onFailure(i);var o={Salt:s.Buffer.from(t.getSaltDevices(),"hex").toString("base64"),PasswordVerifier:s.Buffer.from(t.getVerifierDevices(),"hex").toString("base64")};r.verifierDevices=o.PasswordVerifier,r.deviceGroupKey=a.DeviceGroupKey,r.randomPassword=t.getRandomPassword(),r.client.request("ConfirmDevice",{DeviceKey:a.DeviceKey,AccessToken:r.signInUserSession.getAccessToken().getJwtToken(),DeviceSecretVerifierConfig:o,DeviceName:navigator.userAgent},function(t,i){return t?n.onFailure(t):(r.deviceKey=e.AuthenticationResult.NewDeviceMetadata.DeviceKey,r.cacheDeviceKeyAndPassword(),i.UserConfirmationNecessary===!0?n.onSuccess(r.signInUserSession,i.UserConfirmationNecessary):n.onSuccess(r.signInUserSession))})})},e.prototype.completeNewPasswordChallenge=function(e,t,n){var r=this;if(!e)return n.onFailure(new Error("New password is required."));var i=new l.default(this.pool.getUserPoolId().split("_")[1]),o=i.getNewPasswordRequiredChallengeUserAttributePrefix(),s={};t&&Object.keys(t).forEach(function(e){s[o+e]=t[e]}),s.NEW_PASSWORD=e,s.USERNAME=this.username;var a={ChallengeName:"NEW_PASSWORD_REQUIRED",ClientId:this.pool.getClientId(),ChallengeResponses:s,Session:this.Session};this.getUserContextData()&&(a.UserContextData=this.getUserContextData()),this.client.request("RespondToAuthChallenge",a,function(e,t){return e?n.onFailure(e):r.authenticateUserInternal(t,i,n)})},e.prototype.getDeviceResponse=function(e){var t=this,n=new l.default(this.deviceGroupKey),r=new C.default,i={};i.USERNAME=this.username,i.DEVICE_KEY=this.deviceKey,n.getLargeAValue(function(o,a){o&&e.onFailure(o),i.SRP_A=a.toString(16);var u={ChallengeName:"DEVICE_SRP_AUTH",ClientId:t.pool.getClientId(),ChallengeResponses:i};t.getUserContextData()&&(u.UserContextData=t.getUserContextData()),t.client.request("RespondToAuthChallenge",u,function(i,o){if(i)return e.onFailure(i);var a=o.ChallengeParameters,u=new h.default(a.SRP_B,16),c=new h.default(a.SALT,16);n.getPasswordAuthenticationKey(t.deviceKey,t.randomPassword,u,c,function(n,i){if(n)return e.onFailure(n);var u=r.getNowString(),c=I("sha256",i).update(s.Buffer.concat([s.Buffer.from(t.deviceGroupKey,"utf8"),s.Buffer.from(t.deviceKey,"utf8"),s.Buffer.from(a.SECRET_BLOCK,"base64"),s.Buffer.from(u,"utf8")])).digest("base64"),h={};h.USERNAME=t.username,h.PASSWORD_CLAIM_SECRET_BLOCK=a.SECRET_BLOCK,h.TIMESTAMP=u,h.PASSWORD_CLAIM_SIGNATURE=c,h.DEVICE_KEY=t.deviceKey;var f={ChallengeName:"DEVICE_PASSWORD_VERIFIER",ClientId:t.pool.getClientId(),ChallengeResponses:h,Session:o.Session};t.getUserContextData()&&(f.UserContextData=t.getUserContextData()),t.client.request("RespondToAuthChallenge",f,function(n,r){return n?e.onFailure(n):(t.signInUserSession=t.getCognitoUserSession(r.AuthenticationResult),t.cacheTokens(),e.onSuccess(t.signInUserSession))})})})})},e.prototype.confirmRegistration=function(e,t,n){var r={ClientId:this.pool.getClientId(),ConfirmationCode:e,Username:this.username,ForceAliasCreation:t};this.getUserContextData()&&(r.UserContextData=this.getUserContextData()),this.client.request("ConfirmSignUp",r,function(e){return e?n(e,null):n(null,"SUCCESS")})},e.prototype.sendCustomChallengeAnswer=function(e,t){var n=this,r={};r.USERNAME=this.username,r.ANSWER=e;var i=new l.default(this.pool.getUserPoolId().split("_")[1]);this.getCachedDeviceKeyAndPassword(),null!=this.deviceKey&&(r.DEVICE_KEY=this.deviceKey);var o={ChallengeName:"CUSTOM_CHALLENGE",ChallengeResponses:r,ClientId:this.pool.getClientId(),Session:this.Session};this.getUserContextData()&&(o.UserContextData=this.getUserContextData()),this.client.request("RespondToAuthChallenge",o,function(e,r){return e?t.onFailure(e):n.authenticateUserInternal(r,i,t)})},e.prototype.sendMFACode=function(e,t,n){var r=this,i={};i.USERNAME=this.username,i.SMS_MFA_CODE=e;var o=n||"SMS_MFA";"SOFTWARE_TOKEN_MFA"===o&&(i.SOFTWARE_TOKEN_MFA_CODE=e),null!=this.deviceKey&&(i.DEVICE_KEY=this.deviceKey);var a={ChallengeName:o,ChallengeResponses:i,ClientId:this.pool.getClientId(),Session:this.Session};this.getUserContextData()&&(a.UserContextData=this.getUserContextData()),this.client.request("RespondToAuthChallenge",a,function(e,n){if(e)return t.onFailure(e);var i=n.ChallengeName;if("DEVICE_SRP_AUTH"===i)return void r.getDeviceResponse(t);if(r.signInUserSession=r.getCognitoUserSession(n.AuthenticationResult),r.cacheTokens(),null==n.AuthenticationResult.NewDeviceMetadata)return t.onSuccess(r.signInUserSession);var o=new l.default(r.pool.getUserPoolId().split("_")[1]);o.generateHashDevice(n.AuthenticationResult.NewDeviceMetadata.DeviceGroupKey,n.AuthenticationResult.NewDeviceMetadata.DeviceKey,function(e){if(e)return t.onFailure(e);var i={Salt:s.Buffer.from(o.getSaltDevices(),"hex").toString("base64"),PasswordVerifier:s.Buffer.from(o.getVerifierDevices(),"hex").toString("base64")};r.verifierDevices=i.PasswordVerifier,r.deviceGroupKey=n.AuthenticationResult.NewDeviceMetadata.DeviceGroupKey,r.randomPassword=o.getRandomPassword(),r.client.request("ConfirmDevice",{DeviceKey:n.AuthenticationResult.NewDeviceMetadata.DeviceKey,AccessToken:r.signInUserSession.getAccessToken().getJwtToken(),DeviceSecretVerifierConfig:i,DeviceName:navigator.userAgent},function(e,i){return e?t.onFailure(e):(r.deviceKey=n.AuthenticationResult.NewDeviceMetadata.DeviceKey,r.cacheDeviceKeyAndPassword(),i.UserConfirmationNecessary===!0?t.onSuccess(r.signInUserSession,i.UserConfirmationNecessary):t.onSuccess(r.signInUserSession))})})})},e.prototype.changePassword=function(e,t,n){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("ChangePassword",{PreviousPassword:e,ProposedPassword:t,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(e){return e?n(e,null):n(null,"SUCCESS")}):n(new Error("User is not authenticated"),null)},e.prototype.enableMFA=function(e){if(null==this.signInUserSession||!this.signInUserSession.isValid())return e(new Error("User is not authenticated"),null);var t=[],n={DeliveryMedium:"SMS",AttributeName:"phone_number"};t.push(n),this.client.request("SetUserSettings",{MFAOptions:t,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t){return t?e(t,null):e(null,"SUCCESS")})},e.prototype.setUserMfaPreference=function(e,t,n){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("SetUserMFAPreference",{SMSMfaSettings:e,SoftwareTokenMfaSettings:t,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(e){return e?n(e,null):n(null,"SUCCESS")}):n(new Error("User is not authenticated"),null)},e.prototype.disableMFA=function(e){if(null==this.signInUserSession||!this.signInUserSession.isValid())return e(new Error("User is not authenticated"),null);var t=[];this.client.request("SetUserSettings",{MFAOptions:t,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t){return t?e(t,null):e(null,"SUCCESS")})},e.prototype.deleteUser=function(e){var t=this;return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("DeleteUser",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(n){return n?e(n,null):(t.clearCachedTokens(),e(null,"SUCCESS"))}):e(new Error("User is not authenticated"),null)},e.prototype.updateAttributes=function(e,t){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("UpdateUserAttributes",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),UserAttributes:e},function(e){return e?t(e,null):t(null,"SUCCESS")}):t(new Error("User is not authenticated"),null)},e.prototype.getUserAttributes=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GetUser",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t,n){if(t)return e(t,null);for(var r=[],i=0;i<n.UserAttributes.length;i++){var o={Name:n.UserAttributes[i].Name,Value:n.UserAttributes[i].Value},s=new E.default(o);r.push(s)}return e(null,r)}):e(new Error("User is not authenticated"),null)},e.prototype.getMFAOptions=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GetUser",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t,n){return t?e(t,null):e(null,n.MFAOptions)}):e(new Error("User is not authenticated"),null)},e.prototype.getUserData=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GetUser",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t,n){return t?e(t,null):e(null,n)}):e(new Error("User is not authenticated"),null)},e.prototype.deleteAttributes=function(e,t){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("DeleteUserAttributes",{UserAttributeNames:e,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(e){return e?t(e,null):t(null,"SUCCESS")}):t(new Error("User is not authenticated"),null)},e.prototype.resendConfirmationCode=function(e){var t={ClientId:this.pool.getClientId(),Username:this.username};this.client.request("ResendConfirmationCode",t,function(t,n){return t?e(t,null):e(null,n)})},e.prototype.getSession=function(e){if(null==this.username)return e(new Error("Username is null. Cannot retrieve a new session"),null);if(null!=this.signInUserSession&&this.signInUserSession.isValid())return e(null,this.signInUserSession);var t="CognitoIdentityServiceProvider."+this.pool.getClientId()+"."+this.username,n=t+".idToken",r=t+".accessToken",i=t+".refreshToken",o=t+".clockDrift";if(this.storage.getItem(n)){var s=new v.default({IdToken:this.storage.getItem(n)}),a=new d.default({AccessToken:this.storage.getItem(r)}),u=new m.default({RefreshToken:this.storage.getItem(i)}),c=parseInt(this.storage.getItem(o),0)||0,h={IdToken:s,AccessToken:a,RefreshToken:u,ClockDrift:c},f=new w.default(h);if(f.isValid())return this.signInUserSession=f,e(null,this.signInUserSession);if(null==u.getToken())return e(new Error("Cannot retrieve a new session. Please authenticate."),null);this.refreshSession(u,e)}else e(new Error("Local storage is missing an ID Token, Please authenticate"),null)},e.prototype.refreshSession=function(e,t){var n=this,r={};r.REFRESH_TOKEN=e.getToken();var i="CognitoIdentityServiceProvider."+this.pool.getClientId(),o=i+".LastAuthUser";if(this.storage.getItem(o)){this.username=this.storage.getItem(o);var s=i+"."+this.username+".deviceKey";this.deviceKey=this.storage.getItem(s),r.DEVICE_KEY=this.deviceKey}var a={ClientId:this.pool.getClientId(),AuthFlow:"REFRESH_TOKEN_AUTH",AuthParameters:r};this.getUserContextData()&&(a.UserContextData=this.getUserContextData()),this.client.request("InitiateAuth",a,function(r,i){if(r)return"NotAuthorizedException"===r.code&&n.clearCachedTokens(),t(r,null);if(i){var o=i.AuthenticationResult;return Object.prototype.hasOwnProperty.call(o,"RefreshToken")||(o.RefreshToken=e.getToken()),n.signInUserSession=n.getCognitoUserSession(o),n.cacheTokens(),t(null,n.signInUserSession)}})},e.prototype.cacheTokens=function(){var e="CognitoIdentityServiceProvider."+this.pool.getClientId(),t=e+"."+this.username+".idToken",n=e+"."+this.username+".accessToken",r=e+"."+this.username+".refreshToken",i=e+"."+this.username+".clockDrift",o=e+".LastAuthUser";this.storage.setItem(t,this.signInUserSession.getIdToken().getJwtToken()),this.storage.setItem(n,this.signInUserSession.getAccessToken().getJwtToken()),this.storage.setItem(r,this.signInUserSession.getRefreshToken().getToken()),this.storage.setItem(i,""+this.signInUserSession.getClockDrift()),this.storage.setItem(o,this.username)},e.prototype.cacheDeviceKeyAndPassword=function(){var e="CognitoIdentityServiceProvider."+this.pool.getClientId()+"."+this.username,t=e+".deviceKey",n=e+".randomPasswordKey",r=e+".deviceGroupKey";this.storage.setItem(t,this.deviceKey),this.storage.setItem(n,this.randomPassword),this.storage.setItem(r,this.deviceGroupKey)},e.prototype.getCachedDeviceKeyAndPassword=function(){var e="CognitoIdentityServiceProvider."+this.pool.getClientId()+"."+this.username,t=e+".deviceKey",n=e+".randomPasswordKey",r=e+".deviceGroupKey";this.storage.getItem(t)&&(this.deviceKey=this.storage.getItem(t),this.randomPassword=this.storage.getItem(n),this.deviceGroupKey=this.storage.getItem(r))},e.prototype.clearCachedDeviceKeyAndPassword=function(){var e="CognitoIdentityServiceProvider."+this.pool.getClientId()+"."+this.username,t=e+".deviceKey",n=e+".randomPasswordKey",r=e+".deviceGroupKey";this.storage.removeItem(t),this.storage.removeItem(n),this.storage.removeItem(r)},e.prototype.clearCachedTokens=function(){var e="CognitoIdentityServiceProvider."+this.pool.getClientId(),t=e+"."+this.username+".idToken",n=e+"."+this.username+".accessToken",r=e+"."+this.username+".refreshToken",i=e+".LastAuthUser";this.storage.removeItem(t),this.storage.removeItem(n),this.storage.removeItem(r),this.storage.removeItem(i)},e.prototype.getCognitoUserSession=function(e){var t=new v.default(e),n=new d.default(e),r=new m.default(e),i={IdToken:t,AccessToken:n,RefreshToken:r};return new w.default(i)},e.prototype.forgotPassword=function(e){var t={ClientId:this.pool.getClientId(),Username:this.username};this.getUserContextData()&&(t.UserContextData=this.getUserContextData()),this.client.request("ForgotPassword",t,function(t,n){return t?e.onFailure(t):"function"==typeof e.inputVerificationCode?e.inputVerificationCode(n):e.onSuccess(n)})},e.prototype.confirmPassword=function(e,t,n){var r={ClientId:this.pool.getClientId(),Username:this.username,ConfirmationCode:e,Password:t};this.getUserContextData()&&(r.UserContextData=this.getUserContextData()),this.client.request("ConfirmForgotPassword",r,function(e){return e?n.onFailure(e):n.onSuccess()})},e.prototype.getAttributeVerificationCode=function(e,t){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GetUserAttributeVerificationCode",{AttributeName:e,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(e,n){return e?t.onFailure(e):"function"==typeof t.inputVerificationCode?t.inputVerificationCode(n):t.onSuccess()}):t.onFailure(new Error("User is not authenticated"))},e.prototype.verifyAttribute=function(e,t,n){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("VerifyUserAttribute",{AttributeName:e,Code:t,AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(e){return e?n.onFailure(e):n.onSuccess("SUCCESS")}):n.onFailure(new Error("User is not authenticated"))},e.prototype.getDevice=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GetDevice",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),DeviceKey:this.deviceKey},function(t,n){return t?e.onFailure(t):e.onSuccess(n)}):e.onFailure(new Error("User is not authenticated"))},e.prototype.forgetSpecificDevice=function(e,t){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("ForgetDevice",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),DeviceKey:e},function(e){return e?t.onFailure(e):t.onSuccess("SUCCESS")}):t.onFailure(new Error("User is not authenticated"))},e.prototype.forgetDevice=function(e){var t=this;this.forgetSpecificDevice(this.deviceKey,{onFailure:e.onFailure,onSuccess:function(n){return t.deviceKey=null,t.deviceGroupKey=null,t.randomPassword=null,t.clearCachedDeviceKeyAndPassword(),e.onSuccess(n)}})},e.prototype.setDeviceStatusRemembered=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("UpdateDeviceStatus",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),DeviceKey:this.deviceKey,DeviceRememberedStatus:"remembered"},function(t){return t?e.onFailure(t):e.onSuccess("SUCCESS")}):e.onFailure(new Error("User is not authenticated"))},e.prototype.setDeviceStatusNotRemembered=function(e){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("UpdateDeviceStatus",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),DeviceKey:this.deviceKey,DeviceRememberedStatus:"not_remembered"},function(t){return t?e.onFailure(t):e.onSuccess("SUCCESS")}):e.onFailure(new Error("User is not authenticated"))},e.prototype.listDevices=function(e,t,n){return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("ListDevices",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),Limit:e,PaginationToken:t},function(e,t){return e?n.onFailure(e):n.onSuccess(t)}):n.onFailure(new Error("User is not authenticated"))},e.prototype.globalSignOut=function(e){var t=this;return null!=this.signInUserSession&&this.signInUserSession.isValid()?void this.client.request("GlobalSignOut",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(n){return n?e.onFailure(n):(t.clearCachedTokens(),e.onSuccess("SUCCESS"))}):e.onFailure(new Error("User is not authenticated"))},e.prototype.signOut=function(){this.signInUserSession=null,this.clearCachedTokens()},e.prototype.sendMFASelectionAnswer=function(e,t){var n=this,r={};r.USERNAME=this.username,r.ANSWER=e;var i={ChallengeName:"SELECT_MFA_TYPE",ChallengeResponses:r,ClientId:this.pool.getClientId(),Session:this.Session};this.getUserContextData()&&(i.UserContextData=this.getUserContextData()),this.client.request("RespondToAuthChallenge",i,function(r,i){return r?t.onFailure(r):(n.Session=i.Session,"SMS_MFA"===e?t.mfaRequired(i.challengeName,i.challengeParameters):"SOFTWARE_TOKEN_MFA"===e?t.totpRequired(i.challengeName,i.challengeParameters):void 0)})},e.prototype.getUserContextData=function(){var e=this.pool;return e.getUserContextData(this.username)},e.prototype.associateSoftwareToken=function(e){var t=this;null!=this.signInUserSession&&this.signInUserSession.isValid()?this.client.request("AssociateSoftwareToken",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken()},function(t,n){return t?e.onFailure(t):e.associateSecretCode(n.SecretCode)}):this.client.request("AssociateSoftwareToken",{Session:this.Session},function(n,r){return n?e.onFailure(n):(t.Session=r.Session,e.associateSecretCode(r.SecretCode))})},e.prototype.verifySoftwareToken=function(e,t,n){var r=this;null!=this.signInUserSession&&this.signInUserSession.isValid()?this.client.request("VerifySoftwareToken",{AccessToken:this.signInUserSession.getAccessToken().getJwtToken(),UserCode:e,FriendlyDeviceName:t},function(e,t){return e?n.onFailure(e):n.onSuccess(t)}):this.client.request("VerifySoftwareToken",{Session:this.Session,UserCode:e,FriendlyDeviceName:t},function(e,t){if(e)return n.onFailure(e);r.Session=t.Session;var i={};i.USERNAME=r.username;var o={ChallengeName:"MFA_SETUP",ClientId:r.pool.getClientId(),ChallengeResponses:i,Session:r.Session};r.getUserContextData()&&(o.UserContextData=r.getUserContextData()),r.client.request("RespondToAuthChallenge",o,function(e,t){return e?n.onFailure(e):(r.signInUserSession=r.getCognitoUserSession(t.AuthenticationResult),r.cacheTokens(),n.onSuccess(r.signInUserSession))})})},e}();t.default=R},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r=function(){function e(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},r=t.Name,i=t.Value;n(this,e),this.Name=r||"",this.Value=i||""}return e.prototype.getValue=function(){return this.Value},e.prototype.setValue=function(e){return this.Value=e,this},e.prototype.getName=function(){return this.Name},e.prototype.setName=function(e){return this.Name=e,this},e.prototype.toString=function(){return JSON.stringify(this)},e.prototype.toJSON=function(){return{Name:this.Name,Value:this.Value}},e}();t.default=r},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r=function(){function e(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},r=t.IdToken,i=t.RefreshToken,o=t.AccessToken,s=t.ClockDrift;if(n(this,e),null==o||null==r)throw new Error("Id token and Access Token must be present.");this.idToken=r,this.refreshToken=i,this.accessToken=o,this.clockDrift=void 0===s?this.calculateClockDrift():s}return e.prototype.getIdToken=function(){return this.idToken},e.prototype.getRefreshToken=function(){return this.refreshToken},e.prototype.getAccessToken=function(){return this.accessToken},e.prototype.getClockDrift=function(){return this.clockDrift},e.prototype.calculateClockDrift=function(){var e=Math.floor(new Date/1e3),t=Math.min(this.accessToken.getIssuedAt(),this.idToken.getIssuedAt());return e-t},e.prototype.isValid=function(){var e=Math.floor(new Date/1e3),t=e-this.clockDrift;return t<this.accessToken.getExpiration()&&t<this.idToken.getExpiration()},e}();t.default=r},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],i=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],o=function(){function e(){n(this,e)}return e.prototype.getNowString=function(){var e=new Date,t=i[e.getUTCDay()],n=r[e.getUTCMonth()],o=e.getUTCDate(),s=e.getUTCHours();s<10&&(s="0"+s);var a=e.getUTCMinutes();a<10&&(a="0"+a);var u=e.getUTCSeconds();u<10&&(u="0"+u);var c=e.getUTCFullYear(),h=t+" "+n+" "+o+" "+s+":"+a+":"+u+" UTC "+c;return h},e}();t.default=o},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r={},i=function(){function e(){n(this,e)}return e.setItem=function(e,t){return r[e]=t,r[e]},e.getItem=function(e){return Object.prototype.hasOwnProperty.call(r,e)?r[e]:void 0},e.removeItem=function(e){return delete r[e]},e.clear=function(){return r={}},e}(),o=function(){function e(){n(this,e);try{this.storageWindow=window.localStorage,this.storageWindow.setItem("aws.cognito.test-ls",1),this.storageWindow.removeItem("aws.cognito.test-ls")}catch(e){this.storageWindow=i}}return e.prototype.getStorage=function(){return this.storageWindow},e}();t.default=o},function(e,t,n){function r(e,t,n){a.isBuffer(t)||(t=new a(t)),a.isBuffer(n)||(n=new a(n)),t.length>p?t=e(t):t.length<p&&(t=a.concat([t,d],p));for(var r=new a(p),i=new a(p),o=0;o<p;o++)r[o]=54^t[o],i[o]=92^t[o];var s=e(a.concat([r,n]));return e(a.concat([i,s]))}function i(e,t){e=e||"sha1";var n=l[e],i=[],s=0;return n||o("algorithm:",e,"is not yet supported"),{update:function(e){return a.isBuffer(e)||(e=new a(e)),i.push(e),s+=e.length,this},digest:function(e){var o=a.concat(i),s=t?r(n,t,o):n(o);return i=null,e?s.toString(e):s}}}function o(){var e=[].slice.call(arguments).join(" ");throw new Error([e,"we accept pull requests","http://github.com/dominictarr/crypto-browserify"].join("\n"))}function s(e,t){for(var n in e)t(e[n],n)}var a=n(1).Buffer,u=n(26),c=n(27),h=n(25),f=n(24),l={sha1:u,sha256:c,md5:f},p=64,d=new a(p);d.fill(0),t.createHash=function(e){return i(e)},t.createHmac=function(e,t){return i(e,t)},t.randomBytes=function(e,t){if(!t||!t.call)return new a(h(e));try{t.call(this,void 0,new a(h(e)))}catch(e){t(e)}},s(["createCredentials","createCipher","createCipheriv","createDecipher","createDecipheriv","createSign","createVerify","createDiffieHellman","pbkdf2"],function(e){t[e]=function(){o("sorry,",e,"is not implemented yet")}})},function(e,t){"use strict";function n(e){var t=e.length;if(t%4>0)throw new Error("Invalid string. Length must be a multiple of 4");var n=e.indexOf("=");n===-1&&(n=t);var r=n===t?0:4-n%4;return[n,r]}function r(e){var t=n(e),r=t[0],i=t[1];return 3*(r+i)/4-i}function i(e,t,n){return 3*(t+n)/4-n}function o(e){for(var t,r=n(e),o=r[0],s=r[1],a=new f(i(e,o,s)),u=0,c=s>0?o-4:o,l=0;l<c;l+=4)t=h[e.charCodeAt(l)]<<18|h[e.charCodeAt(l+1)]<<12|h[e.charCodeAt(l+2)]<<6|h[e.charCodeAt(l+3)],a[u++]=t>>16&255,a[u++]=t>>8&255,a[u++]=255&t;return 2===s&&(t=h[e.charCodeAt(l)]<<2|h[e.charCodeAt(l+1)]>>4,a[u++]=255&t),1===s&&(t=h[e.charCodeAt(l)]<<10|h[e.charCodeAt(l+1)]<<4|h[e.charCodeAt(l+2)]>>2,a[u++]=t>>8&255,a[u++]=255&t),a}function s(e){return c[e>>18&63]+c[e>>12&63]+c[e>>6&63]+c[63&e]}function a(e,t,n){for(var r,i=[],o=t;o<n;o+=3)r=(e[o]<<16&16711680)+(e[o+1]<<8&65280)+(255&e[o+2]),i.push(s(r));return i.join("")}function u(e){for(var t,n=e.length,r=n%3,i=[],o=16383,s=0,u=n-r;s<u;s+=o)i.push(a(e,s,s+o>u?u:s+o));return 1===r?(t=e[n-1],i.push(c[t>>2]+c[t<<4&63]+"==")):2===r&&(t=(e[n-2]<<8)+e[n-1],i.push(c[t>>10]+c[t>>4&63]+c[t<<2&63]+"=")),i.join("")}t.byteLength=r,t.toByteArray=o,t.fromByteArray=u;for(var c=[],h=[],f="undefined"!=typeof Uint8Array?Uint8Array:Array,l="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",p=0,d=l.length;p<d;++p)c[p]=l[p],h[l.charCodeAt(p)]=p;h["-".charCodeAt(0)]=62,h["_".charCodeAt(0)]=63},function(e,t){t.read=function(e,t,n,r,i){var o,s,a=8*i-r-1,u=(1<<a)-1,c=u>>1,h=-7,f=n?i-1:0,l=n?-1:1,p=e[t+f];for(f+=l,o=p&(1<<-h)-1,p>>=-h,h+=a;h>0;o=256*o+e[t+f],f+=l,h-=8);for(s=o&(1<<-h)-1,o>>=-h,h+=r;h>0;s=256*s+e[t+f],f+=l,h-=8);if(0===o)o=1-c;else{if(o===u)return s?NaN:(p?-1:1)*(1/0);s+=Math.pow(2,r),o-=c}return(p?-1:1)*s*Math.pow(2,o-r)},t.write=function(e,t,n,r,i,o){var s,a,u,c=8*o-i-1,h=(1<<c)-1,f=h>>1,l=23===i?Math.pow(2,-24)-Math.pow(2,-77):0,p=r?0:o-1,d=r?1:-1,g=t<0||0===t&&1/t<0?1:0;for(t=Math.abs(t),isNaN(t)||t===1/0?(a=isNaN(t)?1:0,s=h):(s=Math.floor(Math.log(t)/Math.LN2),t*(u=Math.pow(2,-s))<1&&(s--,u*=2),t+=s+f>=1?l/u:l*Math.pow(2,1-f),t*u>=2&&(s++,u/=2),s+f>=h?(a=0,s=h):s+f>=1?(a=(t*u-1)*Math.pow(2,i),s+=f):(a=t*Math.pow(2,f-1)*Math.pow(2,i),s=0));i>=8;e[n+p]=255&a,p+=d,a/=256,i-=8);for(s=s<<i|a,c+=i;c>0;e[n+p]=255&s,p+=d,s/=256,c-=8);e[n+p-d]|=128*g}},function(e,t){var n={}.toString;e.exports=Array.isArray||function(e){return"[object Array]"==n.call(e)}},function(e,t,n){var r,i;!function(o){var s=!1;if(r=o,i="function"==typeof r?r.call(t,n,t,e):r,!(void 0!==i&&(e.exports=i)),s=!0,e.exports=o(),s=!0,!s){var a=window.Cookies,u=window.Cookies=o();u.noConflict=function(){return window.Cookies=a,u}}}(function(){function e(){for(var e=0,t={};e<arguments.length;e++){var n=arguments[e];for(var r in n)t[r]=n[r]}return t}function t(n){function r(t,i,o){var s;if("undefined"!=typeof document){if(arguments.length>1){if(o=e({path:"/"},r.defaults,o),"number"==typeof o.expires){var a=new Date;a.setMilliseconds(a.getMilliseconds()+864e5*o.expires),o.expires=a}o.expires=o.expires?o.expires.toUTCString():"";try{s=JSON.stringify(i),/^[\{\[]/.test(s)&&(i=s)}catch(e){}i=n.write?n.write(i,t):encodeURIComponent(String(i)).replace(/%(23|24|26|2B|3A|3C|3E|3D|2F|3F|40|5B|5D|5E|60|7B|7D|7C)/g,decodeURIComponent),t=encodeURIComponent(String(t)),t=t.replace(/%(23|24|26|2B|5E|60|7C)/g,decodeURIComponent),t=t.replace(/[\(\)]/g,escape);var u="";for(var c in o)o[c]&&(u+="; "+c,o[c]!==!0&&(u+="="+o[c]));return document.cookie=t+"="+i+u}t||(s={});for(var h=document.cookie?document.cookie.split("; "):[],f=/(%[0-9A-Z]{2})+/g,l=0;l<h.length;l++){var p=h[l].split("="),d=p.slice(1).join("=");this.json||'"'!==d.charAt(0)||(d=d.slice(1,-1));try{var g=p[0].replace(f,decodeURIComponent);if(d=n.read?n.read(d,g):n(d,g)||d.replace(f,decodeURIComponent),this.json)try{d=JSON.parse(d)}catch(e){}if(t===g){s=d;break}t||(s[g]=d)}catch(e){}}return s}}return r.set=r,r.get=function(e){return r.call(r,e)},r.getJSON=function(){return r.apply({json:!0},[].slice.call(arguments))},r.defaults={},r.remove=function(t,n){r(t,"",e(n,{expires:-1}))},r.withConverter=t,r}return t(function(){})})},function(e,t){"use strict";function n(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;/*!
	 * Copyright 2016 Amazon.com,
	 * Inc. or its affiliates. All Rights Reserved.
	 *
	 * Licensed under the Amazon Software License (the "License").
	 * You may not use this file except in compliance with the
	 * License. A copy of the License is located at
	 *
	 *     http://aws.amazon.com/asl/
	 *
	 * or in the "license" file accompanying this file. This file is
	 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	 * CONDITIONS OF ANY KIND, express or implied. See the License
	 * for the specific language governing permissions and
	 * limitations under the License.
	 */
var r=function(){function e(t){n(this,e);var r=t||{},i=r.ValidationData,o=r.Username,s=r.Password,a=r.AuthParameters;this.validationData=i||{},this.authParameters=a||{},this.username=o,this.password=s}return e.prototype.getUsername=function(){return this.username},e.prototype.getPassword=function(){return this.password},e.prototype.getValidationData=function(){return this.validationData},e.prototype.getAuthParameters=function(){return this.authParameters},e}();t.default=r},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var o=n(23),s=r(o),a=function(){function e(t,n){i(this,e),this.endpoint=n||"https://cognito-idp."+t+".amazonaws.com/",this.userAgent=s.default.prototype.userAgent||"aws-amplify/0.1.x js"}return e.prototype.request=function(e,t,n){var r={"Content-Type":"application/x-amz-json-1.1","X-Amz-Target":"AWSCognitoIdentityProviderService."+e,"X-Amz-User-Agent":this.userAgent},i={headers:r,method:"POST",mode:"cors",cache:"no-cache",body:JSON.stringify(t)},o=void 0;fetch(this.endpoint,i).then(function(e){return o=e,e},function(e){if(e instanceof TypeError)throw new Error("Network error");throw e}).then(function(e){return e.json().catch(function(){return{}})}).then(function(e){if(o.ok)return n(null,e);var t=(e.__type||e.code).split("#").pop(),r={code:t,name:t,message:e.message||e.Message||null};return n(r)}).catch(function(e){var t={code:"UnknownError",message:"Unkown error"};if(o&&o.headers&&o.headers.get("x-amzn-errortype"))try{var r=o.headers.get("x-amzn-errortype").split(":")[0];t={code:r,name:r,statusCode:o.status,message:o.status?o.status.toString():null}}catch(e){return t={code:"UnknownError",message:o.headers.get("x-amzn-errortype")},n(t)}else e instanceof Error&&"Network error"===e.message&&(t={code:"NetworkError",name:e.name,message:e.message});return n(t)})},e}();t.default=a},function(e,t,n){"use strict";function r(e){return e&&e.__esModule?e:{default:e}}function i(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var o=n(20),s=r(o),a=n(9),u=r(a),c=n(13),h=r(c),f=function(){function e(t){i(this,e);var n=t||{},r=n.UserPoolId,o=n.ClientId,a=n.endpoint,u=n.AdvancedSecurityDataCollectionFlag;if(!r||!o)throw new Error("Both UserPoolId and ClientId are required.");if(!/^[\w-]+_.+$/.test(r))throw new Error("Invalid UserPoolId format.");var c=r.split("_")[0];this.userPoolId=r,this.clientId=o,this.client=new s.default(c,a),this.advancedSecurityDataCollectionFlag=u!==!1,this.storage=t.Storage||(new h.default).getStorage()}return e.prototype.getUserPoolId=function(){return this.userPoolId},e.prototype.getClientId=function(){return this.clientId},e.prototype.signUp=function(e,t,n,r,i){var o=this,s={ClientId:this.clientId,Username:e,Password:t,UserAttributes:n,ValidationData:r};this.getUserContextData(e)&&(s.UserContextData=this.getUserContextData(e)),this.client.request("SignUp",s,function(t,n){if(t)return i(t,null);var r={Username:e,Pool:o,Storage:o.storage},s={user:new u.default(r),userConfirmed:n.UserConfirmed,userSub:n.UserSub};return i(null,s)})},e.prototype.getCurrentUser=function(){var e="CognitoIdentityServiceProvider."+this.clientId+".LastAuthUser",t=this.storage.getItem(e);if(t){var n={Username:t,Pool:this,Storage:this.storage};return new u.default(n)}return null},e.prototype.getUserContextData=function(e){if("undefined"!=typeof AmazonCognitoAdvancedSecurityData){var t=AmazonCognitoAdvancedSecurityData;if(this.advancedSecurityDataCollectionFlag){var n=t.getData(e,this.userPoolId,this.clientId);if(n){var r={EncodedData:n};return r}}return{}}},e}();t.default=f},function(e,t,n){"use strict";function r(e){if(e&&e.__esModule)return e;var t={};if(null!=e)for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&(t[n]=e[n]);return t.default=e,t}function i(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}t.__esModule=!0;var o=n(18),s=r(o),a=function(){function e(t){i(this,e),this.domain=t.domain,t.path?this.path=t.path:this.path="/",Object.prototype.hasOwnProperty.call(t,"expires")?this.expires=t.expires:this.expires=365,Object.prototype.hasOwnProperty.call(t,"secure")?this.secure=t.secure:this.secure=!0}return e.prototype.setItem=function(e,t){return s.set(e,t,{path:this.path,expires:this.expires,domain:this.domain,secure:this.secure}),s.get(e)},e.prototype.getItem=function(e){return s.get(e)},e.prototype.removeItem=function(e){return s.remove(e,{path:this.path,domain:this.domain,secure:this.secure})},e.prototype.clear=function(){var e=s.get(),t=void 0;for(t=0;t<e.length;++t)s.remove(e[t]);return{}},e}();t.default=a},function(e,t){"use strict";function n(){}t.__esModule=!0,t.default=n,n.prototype.userAgent="aws-amplify/0.1.x js"},function(e,t,n){function r(e,t){e[t>>5]|=128<<t%32,e[(t+64>>>9<<4)+14]=t;for(var n=1732584193,r=-271733879,i=-1732584194,h=271733878,f=0;f<e.length;f+=16){var l=n,p=r,d=i,g=h;n=o(n,r,i,h,e[f+0],7,-680876936),h=o(h,n,r,i,e[f+1],12,-389564586),i=o(i,h,n,r,e[f+2],17,606105819),r=o(r,i,h,n,e[f+3],22,-1044525330),n=o(n,r,i,h,e[f+4],7,-176418897),h=o(h,n,r,i,e[f+5],12,1200080426),i=o(i,h,n,r,e[f+6],17,-1473231341),r=o(r,i,h,n,e[f+7],22,-45705983),n=o(n,r,i,h,e[f+8],7,1770035416),h=o(h,n,r,i,e[f+9],12,-1958414417),i=o(i,h,n,r,e[f+10],17,-42063),r=o(r,i,h,n,e[f+11],22,-1990404162),n=o(n,r,i,h,e[f+12],7,1804603682),h=o(h,n,r,i,e[f+13],12,-40341101),i=o(i,h,n,r,e[f+14],17,-1502002290),r=o(r,i,h,n,e[f+15],22,1236535329),n=s(n,r,i,h,e[f+1],5,-165796510),h=s(h,n,r,i,e[f+6],9,-1069501632),i=s(i,h,n,r,e[f+11],14,643717713),r=s(r,i,h,n,e[f+0],20,-373897302),n=s(n,r,i,h,e[f+5],5,-701558691),h=s(h,n,r,i,e[f+10],9,38016083),i=s(i,h,n,r,e[f+15],14,-660478335),r=s(r,i,h,n,e[f+4],20,-405537848),n=s(n,r,i,h,e[f+9],5,568446438),h=s(h,n,r,i,e[f+14],9,-1019803690),i=s(i,h,n,r,e[f+3],14,-187363961),r=s(r,i,h,n,e[f+8],20,1163531501),n=s(n,r,i,h,e[f+13],5,-1444681467),h=s(h,n,r,i,e[f+2],9,-51403784),i=s(i,h,n,r,e[f+7],14,1735328473),r=s(r,i,h,n,e[f+12],20,-1926607734),n=a(n,r,i,h,e[f+5],4,-378558),h=a(h,n,r,i,e[f+8],11,-2022574463),i=a(i,h,n,r,e[f+11],16,1839030562),r=a(r,i,h,n,e[f+14],23,-35309556),n=a(n,r,i,h,e[f+1],4,-1530992060),h=a(h,n,r,i,e[f+4],11,1272893353),i=a(i,h,n,r,e[f+7],16,-155497632),r=a(r,i,h,n,e[f+10],23,-1094730640),n=a(n,r,i,h,e[f+13],4,681279174),h=a(h,n,r,i,e[f+0],11,-358537222),i=a(i,h,n,r,e[f+3],16,-722521979),r=a(r,i,h,n,e[f+6],23,76029189),n=a(n,r,i,h,e[f+9],4,-640364487),h=a(h,n,r,i,e[f+12],11,-421815835),i=a(i,h,n,r,e[f+15],16,530742520),r=a(r,i,h,n,e[f+2],23,-995338651),n=u(n,r,i,h,e[f+0],6,-198630844),h=u(h,n,r,i,e[f+7],10,1126891415),i=u(i,h,n,r,e[f+14],15,-1416354905),r=u(r,i,h,n,e[f+5],21,-57434055),n=u(n,r,i,h,e[f+12],6,1700485571),h=u(h,n,r,i,e[f+3],10,-1894986606),i=u(i,h,n,r,e[f+10],15,-1051523),r=u(r,i,h,n,e[f+1],21,-2054922799),n=u(n,r,i,h,e[f+8],6,1873313359),h=u(h,n,r,i,e[f+15],10,-30611744),i=u(i,h,n,r,e[f+6],15,-1560198380),r=u(r,i,h,n,e[f+13],21,1309151649),n=u(n,r,i,h,e[f+4],6,-145523070),h=u(h,n,r,i,e[f+11],10,-1120210379),i=u(i,h,n,r,e[f+2],15,718787259),r=u(r,i,h,n,e[f+9],21,-343485551),n=c(n,l),r=c(r,p),i=c(i,d),h=c(h,g)}return Array(n,r,i,h)}function i(e,t,n,r,i,o){return c(h(c(c(t,e),c(r,o)),i),n)}function o(e,t,n,r,o,s,a){return i(t&n|~t&r,e,t,o,s,a)}function s(e,t,n,r,o,s,a){return i(t&r|n&~r,e,t,o,s,a)}function a(e,t,n,r,o,s,a){return i(t^n^r,e,t,o,s,a)}function u(e,t,n,r,o,s,a){return i(n^(t|~r),e,t,o,s,a)}function c(e,t){var n=(65535&e)+(65535&t),r=(e>>16)+(t>>16)+(n>>16);return r<<16|65535&n}function h(e,t){return e<<t|e>>>32-t}var f=n(2);e.exports=function(e){return f.hash(e,r,16)}},function(e,t){!function(){var t,n,r=this;t=function(e){for(var t,t,n=new Array(e),r=0;r<e;r++)0==(3&r)&&(t=4294967296*Math.random()),n[r]=t>>>((3&r)<<3)&255;return n},r.crypto&&crypto.getRandomValues&&(n=function(e){var t=new Uint8Array(e);return crypto.getRandomValues(t),t}),e.exports=n||t}()},function(e,t,n){function r(e,t){e[t>>5]|=128<<24-t%32,e[(t+64>>9<<4)+15]=t;for(var n=Array(80),r=1732584193,u=-271733879,c=-1732584194,h=271733878,f=-1009589776,l=0;l<e.length;l+=16){for(var p=r,d=u,g=c,v=h,y=f,m=0;m<80;m++){m<16?n[m]=e[l+m]:n[m]=a(n[m-3]^n[m-8]^n[m-14]^n[m-16],1);var S=s(s(a(r,5),i(m,u,c,h)),s(s(f,n[m]),o(m)));f=h,h=c,c=a(u,30),u=r,r=S}r=s(r,p),u=s(u,d),c=s(c,g),h=s(h,v),f=s(f,y)}return Array(r,u,c,h,f)}function i(e,t,n,r){return e<20?t&n|~t&r:e<40?t^n^r:e<60?t&n|t&r|n&r:t^n^r}function o(e){return e<20?1518500249:e<40?1859775393:e<60?-1894007588:-899497514}function s(e,t){var n=(65535&e)+(65535&t),r=(e>>16)+(t>>16)+(n>>16);return r<<16|65535&n}function a(e,t){return e<<t|e>>>32-t}var u=n(2);e.exports=function(e){return u.hash(e,r,20,!0)}},function(e,t,n){var r=n(2),i=function(e,t){var n=(65535&e)+(65535&t),r=(e>>16)+(t>>16)+(n>>16);return r<<16|65535&n},o=function(e,t){return e>>>t|e<<32-t},s=function(e,t){return e>>>t},a=function(e,t,n){return e&t^~e&n},u=function(e,t,n){return e&t^e&n^t&n},c=function(e){return o(e,2)^o(e,13)^o(e,22)},h=function(e){return o(e,6)^o(e,11)^o(e,25)},f=function(e){return o(e,7)^o(e,18)^s(e,3)},l=function(e){return o(e,17)^o(e,19)^s(e,10)},p=function(e,t){var n,r,o,s,p,d,g,v,y,m,S,w,A=new Array(1116352408,1899447441,3049323471,3921009573,961987163,1508970993,2453635748,2870763221,3624381080,310598401,607225278,1426881987,1925078388,2162078206,2614888103,3248222580,3835390401,4022224774,264347078,604807628,770255983,1249150122,1555081692,1996064986,2554220882,2821834349,2952996808,3210313671,3336571891,3584528711,113926993,338241895,666307205,773529912,1294757372,1396182291,1695183700,1986661051,2177026350,2456956037,2730485921,2820302411,3259730800,3345764771,3516065817,3600352804,4094571909,275423344,430227734,506948616,659060556,883997877,958139571,1322822218,1537002063,1747873779,1955562222,2024104815,2227730452,2361852424,2428436474,2756734187,3204031479,3329325298),C=new Array(1779033703,3144134277,1013904242,2773480762,1359893119,2600822924,528734635,1541459225),U=new Array(64);e[t>>5]|=128<<24-t%32,e[(t+64>>9<<4)+15]=t;for(var y=0;y<e.length;y+=16){n=C[0],r=C[1],o=C[2],s=C[3],p=C[4],d=C[5],g=C[6],v=C[7];for(var m=0;m<64;m++)m<16?U[m]=e[m+y]:U[m]=i(i(i(l(U[m-2]),U[m-7]),f(U[m-15])),U[m-16]),S=i(i(i(i(v,h(p)),a(p,d,g)),A[m]),U[m]),w=i(c(n),u(n,r,o)),v=g,g=d,d=p,p=i(s,S),s=o,o=r,r=n,n=i(S,w);C[0]=i(n,C[0]),C[1]=i(r,C[1]),C[2]=i(o,C[2]),C[3]=i(s,C[3]),C[4]=i(p,C[4]),C[5]=i(d,C[5]),C[6]=i(g,C[6]),C[7]=i(v,C[7])}return C};e.exports=function(e){return r.hash(e,p,32,!0)}}])});
//# sourceMappingURL=amazon-cognito-identity.min.js.map

// Taken from https://github.com/sorgerlab/minerva-client-js/blob/06e3c13db13f8b9e5b2c4faef3827c545d34d1bd/index.js
//
const CognitoUser = AmazonCognitoIdentity.CognitoUser;
const CognitoUserPool = AmazonCognitoIdentity.CognitoUserPool;
const AuthenticationDetails = AmazonCognitoIdentity.AuthenticationDetails;

minerva_authenticate = function(username, password) {

  // TODO Perhaps cognitoUser should be a class member and reused everywhere?
  const cognitoUser = new CognitoUser({
    Username: username,
    Pool: new CognitoUserPool({
      UserPoolId : 'us-east-1_YuTF9ST4J',
      ClientId : '6ctsnjjglmtna2q5fgtrjug47k'
    })
  });

  const authenticationDetails = new AuthenticationDetails({
    Username: username,
    Password: password
  });

  const auth = new Promise((resolve, reject) => {
    cognitoUser.authenticateUser(authenticationDetails, {
      onSuccess: result => resolve(result),
      onFailure: err => reject(err),
      mfaRequired: codeDeliveryDetails => reject(codeDeliveryDetails),
      newPasswordRequired: (fields, required) => reject({fields, required})
    });
  });

  return auth
    .then(response => response.getIdToken().getJwtToken());
};


/*
 * Main login script
 */

const use_token = function(token, importImage, get_img_src) {

    // Overwrite importImage function
    FigureModel.prototype.importImage = function(imgDataUrl) {
        var urlList = imgDataUrl.split('/');
        var first_index = urlList.indexOf('imgData');
        var aws_gateway = WEBGATEWAYINDEX.split('/')[2];
        var aws_stage = WEBGATEWAYINDEX.split('/')[3];
        urlList.splice(first_index + 1, 0, token);
        urlList.splice(0, first_index, 'https:', '', aws_gateway, aws_stage);
        arguments[0] = urlList.join('/');
        return importImage.apply(this, arguments);
    };

    // Overwrite get_img_src function
    Panel.prototype.get_img_src = function() {
        var img_src = get_img_src.apply(this, arguments);
        var src_list = img_src.split('/');
        var first_index = src_list.indexOf('render_image');
        src_list.splice(first_index + 1, 0, token);
        var new_src = src_list.join('/');
        return encodeURI(new_src);
    };
};

const login_body = $('#welcomeModal').find('.modal-body');
const login_form = document.createElement('form');
$(login_body).prepend(login_form);

const add_img_form = $('#addImagesModal').find('form');
const add_img_group = $(add_img_form).find('.form-group');

const add_img_button = $(add_img_form).find('button:submit');
$(add_img_button).removeAttr('disabled');

// Hidden unused fake id just to pass integer regex
const enter_fake_id = $(add_img_form).find('.imgIds');
// This allows fake field to pass old integer regex for ids
const fake_id_regex = function() {
    $(add_img_button).removeAttr('disabled');
    $(enter_fake_id).val('0');
    $(enter_fake_id).hide();
};
fake_id_regex();

const enter_uuids = document.createElement('input');
enter_uuids.placeholder = 'Minerva Image UUID';
enter_uuids.id = 'enter_uuids';
enter_uuids.type = 'text';

$(enter_uuids).addClass('form-control');
add_img_group.append(enter_uuids);

const enter_username = document.createElement('input');
enter_username.placeholder = 'Minerva Username';
enter_username.id = 'enter_username';
enter_username.type = 'text';

const enter_password = document.createElement('input');
enter_password.placeholder = 'Minerva Password';
enter_password.id = 'enter_password';
enter_password.type = 'password';

$(enter_username).addClass('form-control');
$(enter_password).addClass('form-control');
login_form.append(enter_username);
login_form.append(enter_password);

// Completely new function to add images
const newAddImages = function(iIds) {
   this.clearSelected();

   // approx work out number of columns to layout new panels
   var paper_width = this.get('paper_width'),
       paper_height = this.get('paper_height'),
       colCount = Math.ceil(Math.sqrt(iIds.length)),
       rowCount = Math.ceil(iIds.length/colCount),
       centre = {x: paper_width/2, y: paper_height/2},
       px, py, spacer, scale,
       coords = {'px': px,
                 'py': py,
                 'c': centre,
                 'spacer': spacer,
                 'colCount': colCount,
                 'rowCount': rowCount,
                 'paper_width': paper_width};

   var invalidIds = [];
   for (var i=0; i<iIds.length; i++) {
       var imgId = iIds[i],
           re = /^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$/;
       if (!re.test(imgId)) {
           invalidIds.push(imgId);
       } else {
           imgDataUrl = BASE_WEBFIGURE_URL + 'imgData/' + imgId + '/';
           this.importImage(imgDataUrl, coords, undefined, i);
       }
   }
   if (invalidIds.length > 0) {
       var plural = invalidIds.length > 1 ? "s" : "";
       alert("Could not add image with invalid ID" + plural + ": " + invalidIds.join(", "));
   }
};

// Overwrite AddImagesModalView submit event
(function(addImages, load_from_OMERO, importImage, get_img_src) {
    FigureModel.prototype.addImages = function() {

        const uuids = $(enter_uuids).val();
        const username = $(enter_username).val();
        const password = $(enter_password).val();

        const THIS = this;

        minerva_authenticate(username, password).then(function(token) {
            use_token(token, importImage, get_img_src);
            fake_id_regex();
            addImages.call(THIS, uuids.split(','));
        }).catch(function(e) {
            alert('Invalid Minerva Username or Password');
            fake_id_regex();
        });
    };

    FigureModel.prototype.load_from_OMERO = function(fileId, success) {

        const username = $(enter_username).val();
        const password = $(enter_password).val();

        const THIS = this;
        const ARGS = arguments;

        minerva_authenticate(username, password).then(function(token) {
            use_token(token, importImage, get_img_src);
            load_from_OMERO.apply(THIS, ARGS);
        }).catch(function(e) {
            alert('Invalid Minerva Username or Password');
        });
    }
})(
    newAddImages,
    FigureModel.prototype.load_from_OMERO,
    FigureModel.prototype.importImage,
    Panel.prototype.get_img_src
);
'''


def _event_path_param(event, key):
    '''
    return {
        'token': 'eyJraWQiOiJYT0E0b01xV1RsMzFBbGRMQUh3UXNzREoyWEg5ZnFlU015MVJaVXdSb2dvPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI2Mjk2MmYzYy03OTI0LTRlODctYThmNS02NjY4OTEyMTlhZjUiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLnVzLWVhc3QtMS5hbWF6b25hd3MuY29tXC91cy1lYXN0LTFfWXVURjlTVDRKIiwiY29nbml0bzp1c2VybmFtZSI6IjYyOTYyZjNjLTc5MjQtNGU4Ny1hOGY1LTY2Njg5MTIxOWFmNSIsInByZWZlcnJlZF91c2VybmFtZSI6Imd1ZXN0IiwiYXVkIjoiNmN0c25qamdsbXRuYTJxNWZndHJqdWc0N2siLCJldmVudF9pZCI6IjNhM2FjNGQxLWMzMmItMTFlOC1iOGRhLWViMDA0MDViZGQ5ZCIsInRva2VuX3VzZSI6ImlkIiwiYXV0aF90aW1lIjoxNTM4MTQ1MTA2LCJuYW1lIjoiTWluZXJ2YSBHdWVzdCIsImV4cCI6MTUzODE0ODcwNiwiaWF0IjoxNTM4MTQ1MTA2LCJlbWFpbCI6ImpvaG5AaG9mZi5pbiJ9.aXj6moXfi4xvuCA45r7NWYR0wqYDJK1hKhNYzqRLzc-g4zeOpaqUH5lkFMnSqDKuownr9o0PrPZ6NQEb74j3miDT1vjbeQdFsPwPS-HNvJwuQ1D_khzF4GbUigaec5cE9gYtpin3TSwkUBeVetYP7mDupggKBi996BzKNZN8D6yL5TUeOduyu453qWqU91idxKUkCUeIOyuvtR1dzU-6zvAM6rV_reeTyLfyfth_lC6y4Jpn4r3oRBT3-2cHMc6TmCgqcPpVCPROfp7ojWNNXQshHqf3iyRxo_MOQP3rGmVexxDL9r4p571LHcaJ8UqOG9sy20S6pfMXXWqvf1qLRA',
        'uuid': 'afd6f4bd-67de-4df2-b518-0e9b05a49012',
        'z': '0',
        't': '0'
    }[key]
    '''
    return event['pathParameters'][key]


def _event_query_params(event):
    '''
    return {
        'c': '1|0:21627$FFFFFF',
        'maps': '[{"reverse":{"enabled":false}}]',
        'm': 'c'
    }
    '''
    return event['queryStringParameters']


def json_custom(obj: Any) -> str:
    '''JSON serializer for extra types.
    '''

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type {} not serializable".format(type(obj)))


def _s3_get(client, bucket, uuid, x, y, z, t, c, level):
    '''Fetch a specific PNG from S3 and decode'''

    # Use the grid to build the key
    key = f'{uuid}/C{c}-T{t}-Z{z}-L{level}-Y{y}-X{x}.png'

    obj = boto3.resource('s3').Object(bucket, key)
    body = obj.get()['Body']
    data = body.read()
    stream = BytesIO(data)
    image = cv2.imdecode(np.fromstring(stream.getvalue(),
                                       dtype=np.uint8), 0)
    return image


def make_response(code, body):
    '''Build a response.
        Args:
            code: HTTP response code.
            body: String
        Returns:
            Response object compatible with AWS Lambda Proxy Integration
    '''

    return {
        'statusCode': code,
        'headers': {
            'Content-Type': 'text/plain',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': body
    }


def make_json_response(code: int, body: Union[Dict, List]) -> Dict[str, Any]:
    '''Build a response.
        Args:
            code: HTTP response code.
            body: Python dictionary or list to jsonify.
        Returns:
            Response object compatible with AWS Lambda Proxy Integration
    '''

    return {
        'statusCode': code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': json.dumps(body, default=json_custom)
    }


def make_binary_response(code: int, body: np.ndarray) -> Dict[str, Any]:
    '''Build a binary response.
        Args:
            code: HTTP response code.
            body: Numpy array representing image.
        Returns:
            Response object compatible with AWS Lambda Proxy Integration
    '''

    response = {
        'statusCode': code,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': base64.b64encode(body).decode('utf-8'),
        'isBase64Encoded': True
    }
    return response


def _event_body(event):
    if 'body' in event and event['body'] is not None:
        return json.loads(event['body'])
    return {}


def response(code: int) -> Callable[..., Dict[str, Any]]:
    '''Decorator for turning exceptions into responses.
    KeyErrors are assumed to be missing parameters (either query or path) and
    mapped to 400.
    ValueErrors are assumed to be parameters (either query or path) that fail
    validation and mapped to 422.
    Any other Exceptions are unknown and mapped to 500.
    Args:
        code: HTTP status code.
    Returns:
        Function which returns a response object compatible with AWS Lambda
        Proxy Integration.
    '''

    def wrapper(fn):
        @wraps(fn)
        def wrapped(self, event, context):

            # Execute the function and make a response or error response
            try:
                self.body = _event_body(event)
                output = fn(self, event, context)
                output_type = type(output)

                # Return binary, json, or text
                if output_type is np.ndarray:
                    return make_binary_response(code, output)
                elif output_type in (dict, list):
                    return make_json_response(code, output)
                else:
                    return make_response(code, str(output))
            except KeyError as e:
                logger.exception(e)
                return make_json_response(400, {'error': str(e)})
            except ValueError as e:
                logger.exception(e)
                return make_json_response(422, {'error': str(e)})
            except Exception as e:
                logger.exception(e)
                return make_json_response(500, {'error': str(e)})

        return wrapped
    return wrapper


class Handler:

    token = None
    uuid = None
    z = 0
    t = 0
    bucket = 'minerva-test-cf-common-tilebucket-1su418jflefem'
    # Domain of Minerva
    domain = 'lze4t3ladb.execute-api.us-east-1.amazonaws.com/dev'
    # Domain of the Minerva-OMERO adpater
    own_domain = 'nwwo7xr274.execute-api.us-east-1.amazonaws.com/beta'

    def load_tile(self, c, l, settings):
        ''' Minerva loads a single tile
        
        Args:
            c: channel id
            l: pyramid level
            settings: dict like {
                'grid': (y, x),
                'color': (red, green, blue),
                'min': 0,
                'max': 1
            }

        Returns:
            Modified settings with loaded image
        '''

        client = None
        uuid = self.uuid
        bucket = self.bucket
        (y, x) = settings['grid']

        args = (client, bucket, uuid, x, y, 0, 0, c, l)
        settings['image'] = _s3_get(*args)
        return settings

    def do_crop(self, channels, tile_shape, full_origin, full_shape,
                levels=1, max_size=2000):
        ''' Interface with minerva_lib.crop

        Args:
            channels: List of dicts of channel rendering settings
            tile_shape: The height, width of a single tile
            full_origin: Request's full-resolution y, x origin
            full_shape: Request's full-resolution height, width
            levels: The number of pyramid levels
            max_size: The maximum response height or width

        Returns:
            2D numpy float array of with height, width of at most max_size
            The array is a composite of all channels for full or partial
            tiles within `full_shape` from `full_origin`.
        '''

        level = crop.get_optimum_pyramid_level(full_shape, levels,
                                               max_size, False)

        # Select all tiles for any given channel at pyramid level
        crop_size = crop.transform_coordinates_to_level(full_shape, level)
        crop_origin = crop.transform_coordinates_to_level(full_origin, level)
        tiles = crop.select_grids(tile_shape, crop_origin, crop_size)
        print(f'Cropping 1/{level} scale')

        load_args = []

        for channel in channels:

            (red, green, blue) = channel['color']
            _id = channel['channel']
            _min = channel['min']
            _max = channel['max']

            for (y, x) in tiles:

                load_args.append((_id, level, {
                    'min': _min,
                    'max': _max,
                    'grid': (y, x),
                    'color': (red, green, blue),
                }))

        thread_count = len(channels) * len(tiles)
        pool = ThreadPool(processes=thread_count)
        image_tiles = pool.starmap(self.load_tile, load_args)

        return crop.composite_subtiles(image_tiles, tile_shape,
                                       crop_origin, crop_size)

    @response(200)
    def open_with(self, event, context):

        root = 'https://' + self.own_domain

        return {
          'open_with_options': [{
            'script_url': f'{root}/webgateway/open_with/minerva_login.js'
          }]
        }

    @response(200)
    def open_with_minerva_login(self, event, context):
        
        return MINERVA_LOGIN

    @response(200)
    def image_data(self, event, context):

        self.token = _event_path_param(event, 'token')
        self.uuid = _event_path_param(event, 'uuid')

        return MinervaApi.load_config(self.uuid, self.token,
                                      self.bucket, self.domain)

    @response(200)
    def render(self, event, context):
        '''Render the specified tile with the given settings'''

        # Path and Query parameters
        query_dict = _event_query_params(event)
        self.token = _event_path_param(event, 'token')
        self.uuid = _event_path_param(event, 'uuid')
        self.z = _event_path_param(event, 'z')
        self.t = _event_path_param(event, 't')

        keys = OmeroApi.scaled_region([self.uuid, self.z, self.t],
                                      query_dict, self.token,
                                      self.bucket, self.domain)

        # Make array of channel parameters
        inputs = zip(keys['chan'], keys['c'], keys['r'])
        channels = list(map(MinervaApi.format_input, inputs))

        # Region with margins
        outer_origin = keys['origin']
        outer_shape = keys['shape']
        outer_end = np.array(outer_origin) + outer_shape
        out_h, out_w = outer_shape.astype(np.int64)
        out = np.ones((out_h, out_w, 3)) * 0.5

        # Actual image content
        image_shape = keys['image_shape']
        request_origin = np.maximum(outer_origin, 0)
        request_end = np.minimum(outer_end, image_shape)
        request_shape = request_end - request_origin

        # Minerva does the cropping
        image = self.do_crop(channels, keys['tile_shape'], request_origin,
                             request_shape, keys['levels'], keys['max_size'])

        # Position cropped region within margins
        y_0, x_0 = (request_origin - outer_origin).astype(np.int64)
        y_1, x_1 = [y_0, x_0] + request_shape
        out[y_0:y_1, x_0:x_1] = image

        # Return encoded png
        output = (255 * out).astype(np.uint8)[:, :, ::-1]
        png_output = cv2.imencode('.png', output)[1]
        print('encoded length is ', len(png_output))
        return png_output


handler = Handler()
render = handler.render
open_with = handler.open_with
image_data = handler.image_data
open_with_minerva_login = handler.open_with_minerva_login
