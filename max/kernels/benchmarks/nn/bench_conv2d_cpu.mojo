# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Compare direct and dispatched CPU convolution on detector shapes.

Includes filter conversion and scratch allocation in dispatch. Set
VECLIB_MAXIMUM_THREADS=1 to restrict Accelerate; direct convolution uses
the runtime worker pool, whose size is printed before measurements.
"""

from std.benchmark import Bench, BenchConfig, BenchId, Bencher, keep
from std.math import ceildiv
from std.random import rand
from std.runtime import parallelism_level
from std.sys import simd_width_of
from layout import Coord, TileTensor, row_major
from nn.conv.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    conv_nhwc_direct,
    pack_filter,
)
from nn.conv.conv_utils import (
    get_direct_conv_micro_kernel_width,
    get_conv_shape,
)
from std.utils.index import Index


def bench_conv2d(
    mut bench: Bench, n: Int, h: Int, w: Int, c: Int, f: Int, stride: Int
) raises:
    var ho = (h - 1) // stride + 1
    var wo = (w - 1) // stride + 1
    comptime block = get_direct_conv_micro_kernel_width() * simd_width_of[
        DType.float32
    ]()
    var inputs = List(length=n * h * w * c, fill=Float32(0))
    var filters = List(length=9 * c * f, fill=Float32(0))
    var packed_filters = List(
        length=ceildiv(f, block) * 9 * c * block, fill=Float32(0)
    )
    var outputs = List(length=n * ho * wo * f, fill=Float32(0))
    rand(inputs)
    rand(filters)
    var input = TileTensor(inputs, row_major((n, h, w, c))).as_imm()
    var filter = TileTensor(filters, row_major((3, 3, c, f))).as_imm()
    var packed = TileTensor(
        packed_filters, row_major((ceildiv(f, block), 3, 3, c, block))
    )
    pack_filter(filter, packed, 1)
    var weights = packed.as_imm()
    var output = TileTensor(outputs, row_major((n, ho, wo, f)))
    var shape = get_conv_shape[2, True](
        output,
        input,
        weights,
        Coord(Index(stride, stride)),
        Coord(Index(1, 1)),
        Coord(Index(0, 0)),
        Coord(Index(1, 1)),
        Coord(Index(1, 1)),
        1,
    )

    @inline(.always)
    def no_epilogue[
        dtype: DType, width: SIMDLength, alignment: Int = 1
    ](coords: Coord, value: SIMD[dtype, width]):
        pass

    def direct(mut b: Bencher) raises {imm}:
        def run() raises {imm}:
            ConvDirectNHWC[
                input.LayoutType,
                weights.LayoutType,
                output.LayoutType,
                DType.float32,
                DType.float32,
                DType.float32,
                True,
                ConvInfoStatic[2](),
            ].run(output, input, weights, shape)
            keep(output.ptr)

        b.iter(run)

    def dispatched(mut b: Bencher) raises {imm}:
        def run() raises {imm}:
            conv_nhwc_direct[
                DType.float32,
                DType.float32,
                DType.float32,
                True,
                ConvInfoStatic[2](),
                False,
            ](
                input,
                weights,
                output,
                Index(stride, stride),
                Index(1, 1),
                Index(0, 0),
                Index(1, 1),
                Index(1, 1),
                1,
                no_epilogue,
            )
            keep(output.ptr)

        b.iter(run)

    var label = String(n, "x", h, "x", w, "x", c, "->", f, ";stride=", stride)
    bench.bench_function(direct, BenchId("direct", label))
    bench.bench_function(dispatched, BenchId("dispatch", label))


def main() raises:
    print("Runtime workers:", parallelism_level())
    var bench = Bench(BenchConfig())
    bench_conv2d(bench, 1, 640, 640, 3, 16, 2)
    bench_conv2d(bench, 1, 320, 320, 16, 32, 2)
    bench_conv2d(bench, 1, 80, 80, 64, 64, 1)
    bench_conv2d(bench, 1, 80, 80, 128, 128, 2)
    bench.dump_report()
