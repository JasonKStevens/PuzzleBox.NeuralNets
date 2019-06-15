using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using PuzzleBox.NeuralNets.Algebra;

namespace PuzzleBox.NeuralNets.Test.Algebra
{
    public class MathNetExtensionsFixture
    {
        [Test]
        public void should_perform_simple_convolve()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2 },
                { 3, 4 },
            }.ToMatrix();

            var g = new float[,] {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 },
            }.ToMatrix();

            var expected = new float[,] {
                { 1*1 + 2*2 + 3*4 + 4*5, 1*2 + 2*3 + 3*5 + 4*6 },
                { 1*4 + 2*5 + 3*7 + 4*8, 1*5 + 2*6 + 3*8 + 4*9 },
            }.ToMatrix();

            // Act
            var h = f.Convolve(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_simple_transpose_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 4, 3 },
                { 2, 1 },
            }.ToMatrix();

            var g = new float[,] {
                { 5, 6 },
                { 7, 8 },
            }.ToMatrix();

            var expected = new float[,] {
                { 4*5, 3*5 + 4*6, 3*6 },
                { 2*5 + 4*7, 1*5 + 2*6 + 3*7 + 4*8, 1*6 + 3*8 },
                { 2*7, 1*7 + 2*8, 1*8 },
            }.ToMatrix();

            // Act
            var h = f.ConvolveTranspose(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_strided_1d_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 3, 4 },
            }.ToMatrix();

            var g = new float[,] {
                { 1, 2, 3, 4, 5, 6, 7 },
            }.ToMatrix();

            var expected = new float[,] {
                { 3*1 + 4*2, 3*4 + 4*5 },
            }.ToMatrix();

            // Act
            var h = f.Convolve(g, columnStride: 3);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_minimal_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 4 },
            }.ToMatrix();

            var g = new float[,] {
                { 3 }
            }.ToMatrix();

            var expected = new float[,] {
                { 3*4 },
            }.ToMatrix();

            // Act
            var h = f.Convolve(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_full_padded_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 0.41f, -0.14f },
                { -0.23f, 0.63f }
            }.ToMatrix();

            var g = new float[,] {
                { 0.0594f, 1.1206f },
                { 1.8499f, 0.8183f },
            }.ToMatrix();

            var expected = new float[,] {
                { 0.037422f, 0.692316f, -0.257738f },
                { 1.157121f, -0.042478f, 0.271237f },
                { -0.258986f, 0.643897f, 0.335503f },
            }.ToMatrix();

            // Act
            var h = f.Convolve(g, rowPadding: 1, columnPadding: 1);

            // Assert
            Assert.That(h.AlmostEqual(expected, 0.0001f), Is.True);
        }

        [Test]
        public void should_perform_simple_1d_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 3, 4 },
            }.ToMatrix();

            var g = new float[,] {
                { 1, 2, 3 },
            }.ToMatrix();

            var expected = new float[,] {
                { 3*1 + 4*2, 3*2 + 4*3 },
            }.ToMatrix();

            // Act
            var h = f.Convolve(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_1d_transpose_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2, 3, 4 },
            }.ToMatrix();

            var g = new float[,] {
                { 1 },
            }.ToMatrix();

            var expected = new float[,] {
                { 1, 2, 3, 4 },
            }.ToMatrix();

            // Act
            var h = f.ConvolveTranspose(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_perform_simple_2d_transpose_convolution()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2 },
                { 3, 4 }
            }.ToMatrix();

            var g = new float[,] {
                { 1 },
            }.ToMatrix();

            var expected = new float[,] {
                { 1, 2 },
                { 3, 4 },
            }.ToMatrix();

            // Act
            var h = f.ConvolveTranspose(g);

            // Assert
            Assert.That(h, Is.EqualTo(expected));
        }

        [Test]
        public void should_rotate_matrix_180_degrees()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2 },
                { 3, 4 }
            }.ToMatrix();

            var expected = new float[,] {
                { 4, 3 },
                { 2, 1 }
            }.ToMatrix();

            // Act
            var g = f.Rotate180();

            // Assert
            Assert.That(g, Is.EqualTo(expected));
        }

        [Test]
        public void should_split_by_column()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2, 5, 5, 6, 7 },
                { 3, 4, 5, 5, 8, 9 }
            }.ToMatrix();

            var expected = new Matrix<float>[] {
                new float[,] {
                    { 1, 2 },
                    { 3, 4 }
                }.ToMatrix(),
                new float[,] {
                    { 5, 5 },
                    { 5, 5 }
                }.ToMatrix(),
                new float[,] {
                    { 6, 7 },
                    { 8, 9 }
                }.ToMatrix(),
            };

            // Act
            var g = f.SplitByColumn(3);

            // Assert
            Assert.That(g, Is.EqualTo(expected));
        }

        [Test]
        public void should_throw_when_partial_split_by_column()
        {
            // Arrange
            var f = new float[,] {
                { 1, 2, 5 },
                { 3, 4, 5 }
            }.ToMatrix();

            var expected = new Matrix<float>[] {
                new float[,] {
                    { 1, 2 },
                    { 3, 4 }
                }.ToMatrix(),
                new float[,] {
                    { 5, 5 },
                    { 5, 5 }
                }.ToMatrix(),
            };

            // Act & Assert
            Assert.That(() => f.SplitByColumn(2), Throws.Exception);
        }
    }
}
