from mpmath import mp, matrix, sqrt, sqrtm, exp, pi, erf, powm, det

# Set the precision for mpmath
mp.dps = 50


def mvn_cdf(mean, covariance_matrix, point):
    print(point)
    print(mean)
    print(covariance_matrix)
    """
    Compute the multivariate normal CDF using mpmath.

    Parameters:
    - mean: Mean vector (mpmath matrix).
    - covariance_matrix: Covariance matrix (mpmath matrix).
    - point: Point in the multivariate space (mpmath matrix).

    Returns:
    - CDF value (mpmath mpf).
    """
    dim = len(mean)

    # Compute the Mahalanobis distance
    diff = point - mean
    inv_cov = powm(covariance_matrix, -1)
    mahalanobis_distance = sqrtm(diff.transpose() * inv_cov * diff)

    # Compute the CDF using the complementary error function (erfc)
    cdf_value = 0.5 * (1 + erf(mahalanobis_distance[0] / sqrt(2)))

    return cdf_value


def mvn_pdf(mean, covariance_matrix, point):
    """
    Compute the multivariate normal PDF using mpmath.

    Parameters:
    - mean: Mean vector (mpmath matrix).
    - covariance_matrix: Covariance matrix (mpmath matrix).
    - point: Point in the multivariate space (mpmath matrix).

    Returns:
    - PDF value (mpmath mpf).
    """
    dim = len(mean)

    # Compute the Mahalanobis distance
    diff = point - mean
    inv_cov = powm(covariance_matrix, -1)
    mahalanobis_distance = sqrtm(diff.transpose() * inv_cov * diff)

    # Compute the normalization constant
    norm_const = 1 / sqrt((2 * pi) ** dim * det(covariance_matrix))

    # Compute the PDF value
    pdf_value = norm_const * exp(-0.5 * mahalanobis_distance[0] ** 2)

    return pdf_value


if __name__ == "__main__":
    # Example usage:

    mean_vector = matrix(
        [
            2.77254056,
            1.0,
            0.74081521,
            -1.09070023,
            -0.23520732,
            1.3738802,
            0.93660851,
            1.70753914,
            2.6620536,
            2.13423941,
            4.09327243,
            -1.0,
        ]
    )
    covariance_matrix = matrix(
        [
            [
                1.60774637e01,
                -9.10462790e-01,
                2.06953301e-01,
                -8.88178420e-16,
                3.70208873e00,
                5.72567950e00,
                9.07059930e00,
                1.05480213e01,
                1.15342441e01,
                1.37073390e01,
                1.32697932e01,
                -3.50824981e00,
            ],
            [
                -9.10462790e-01,
                1.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -4.27200000e-01,
                2.59023743e-01,
                -8.63111272e-01,
                -6.40805956e-01,
                -1.12318797e-01,
                -9.89607473e-01,
                6.61483735e-01,
                0.00000000e00,
            ],
            [
                2.06953301e-01,
                0.00000000e00,
                2.50982235e-02,
                0.00000000e00,
                -2.73746383e-02,
                -2.61828342e-02,
                1.50853107e-02,
                3.46826374e-02,
                6.85496242e-02,
                1.17595420e-01,
                1.40682094e-01,
                0.00000000e00,
            ],
            [
                -8.88178420e-16,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                5.55111512e-17,
                0.00000000e00,
                0.00000000e00,
                4.44089210e-16,
                0.00000000e00,
                -4.44089210e-16,
                0.00000000e00,
            ],
            [
                3.70208873e00,
                -4.27200000e-01,
                -2.73746383e-02,
                0.00000000e00,
                1.46235736e00,
                1.84392169e00,
                2.77909176e00,
                3.03375220e00,
                3.04588922e00,
                3.57078238e00,
                2.99109274e00,
                -1.00000000e00,
            ],
            [
                5.72567950e00,
                2.59023743e-01,
                -2.61828342e-02,
                5.55111512e-17,
                1.84392169e00,
                3.54949894e00,
                4.06109587e00,
                4.72445109e00,
                5.28992241e00,
                5.35519095e00,
                6.01337618e00,
                -1.74081521e00,
            ],
            [
                9.07059930e00,
                -8.63111272e-01,
                1.50853107e-02,
                0.00000000e00,
                2.77909176e00,
                4.06109587e00,
                6.56922342e00,
                7.17154038e00,
                7.31537578e00,
                8.53144597e00,
                7.43834798e00,
                -2.28962238e00,
            ],
            [
                1.05480213e01,
                -6.40805956e-01,
                3.46826374e-02,
                0.00000000e00,
                3.03375220e00,
                4.72445109e00,
                7.17154038e00,
                8.45745783e00,
                8.78859434e00,
                9.88006497e00,
                9.19584253e00,
                -2.69618708e00,
            ],
            [
                1.15342441e01,
                -1.12318797e-01,
                6.85496242e-02,
                4.44089210e-16,
                3.04588922e00,
                5.28992241e00,
                7.31537578e00,
                8.78859434e00,
                1.00479176e01,
                1.07253151e01,
                1.09325144e01,
                -2.99737639e00,
            ],
            [
                1.37073390e01,
                -9.89607473e-01,
                1.17595420e-01,
                0.00000000e00,
                3.57078238e00,
                5.35519095e00,
                8.53144597e00,
                9.88006497e00,
                1.07253151e01,
                1.28828880e01,
                1.16352116e01,
                -3.22050201e00,
            ],
            [
                1.32697932e01,
                6.61483735e-01,
                1.40682094e-01,
                -4.44089210e-16,
                2.99109274e00,
                6.01337618e00,
                7.43834798e00,
                9.19584253e00,
                1.09325144e01,
                1.16352116e01,
                1.38152593e01,
                -3.38579687e00,
            ],
            [
                -3.50824981e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -1.00000000e00,
                -1.74081521e00,
                -2.28962238e00,
                -2.69618708e00,
                -2.99737639e00,
                -3.22050201e00,
                -3.38579687e00,
                1.00000000e00,
            ],
        ]
    )
    point_in_space = matrix(
        [
            1.0e10,
            -1.0e10,
            -1.0e10,
            -1.0e10,
            -1.0e10,
            -1.0e10,
            1.0e10,
            1.0e10,
            -1.0e10,
            1.0e10,
            -1.0e10,
            -1.0e10,
        ]
    )

    cdf_result = mvn_cdf(mean_vector, covariance_matrix, point_in_space)
    pdf_result = mvn_pdf(mean_vector, covariance_matrix, point_in_space)
    print(f"Multivariate Gaussian CDF using mpmath:{float(pdf_result)}")
