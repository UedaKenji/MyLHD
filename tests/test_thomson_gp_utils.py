from __future__ import annotations

import io

import numpy as np
import pytest

from mylhd.thomson_gp.utils import Logger, numerical_differentiation_matrix, write_profile


def test_numerical_differentiation_matrix_equal_spacing() -> None:
    x = np.linspace(0.0, 1.0, 5)
    matrix = numerical_differentiation_matrix(x)
    h = x[1] - x[0]

    expected = np.zeros_like(matrix)
    expected[0, 0] = -1.0 / h
    expected[0, 1] = 1.0 / h
    expected[-1, -2] = -1.0 / h
    expected[-1, -1] = 1.0 / h
    expected[1, 0] = -1.0 / (2.0 * h)
    expected[1, 2] = 1.0 / (2.0 * h)
    expected[2, 1] = -1.0 / (2.0 * h)
    expected[2, 3] = 1.0 / (2.0 * h)
    expected[3, 2] = -1.0 / (2.0 * h)
    expected[3, 4] = 1.0 / (2.0 * h)

    np.testing.assert_allclose(matrix, expected)


def test_numerical_differentiation_matrix_non_uniform_spacing() -> None:
    x = np.array([0.0, 0.3, 0.6, 1.0])
    matrix = numerical_differentiation_matrix(x)

    expected = np.zeros_like(matrix)
    expected[0, 0] = -1.0 / 0.3
    expected[0, 1] = 1.0 / 0.3
    expected[1, 0] = -1.0 / 0.6
    expected[1, 2] = 1.0 / 0.6
    expected[2, 1] = -1.0 / 0.7
    expected[2, 3] = 1.0 / 0.7
    expected[3, 2] = -1.0 / 0.4
    expected[3, 3] = 1.0 / 0.4

    np.testing.assert_allclose(matrix, expected)


def test_write_profile_creates_expected_file(tmp_path) -> None:
    filename = tmp_path / "profile.txt"
    reff = np.array([0.1, 0.2])
    ne = np.array([1.5, 2.0])
    te = np.array([2.5, 3.0])
    zeff = np.array([1.0, 1.2])

    write_profile(filename, reff, ne, te, zeff)

    content = filename.read_text().splitlines()
    assert content[0] == "CC  reff/a     Ne[1/m^3]     Te[keV]     Zeff"
    assert content[1] == " Number_of_points   2"
    assert content[2] == f" {reff[0]:10.4E}  {(ne[0] * 1e19):10.4E}  {te[0]:10.4E}  {zeff[0]:10.4E}"
    assert content[3] == f" {reff[1]:10.4E}  {(ne[1] * 1e19):10.4E}  {te[1]:10.4E}  {zeff[1]:10.4E}"


def test_write_profile_raises_on_length_mismatch(tmp_path) -> None:
    filename = tmp_path / "invalid.txt"
    reff = np.array([0.1, 0.2])
    ne = np.array([1.0])
    te = np.array([2.5, 3.0])
    zeff = np.array([1.0, 1.2])

    with pytest.raises(ValueError):
        write_profile(filename, reff, ne, te, zeff)


def test_logger_records_written_messages() -> None:
    buffer = io.StringIO()
    logger = Logger()
    logger.terminal = buffer

    logger.write("Hello, world!\n")
    logger.write("Second line.")
    logger.flush()

    assert buffer.getvalue() == "Hello, world!\nSecond line."
    assert logger.log == "Hello, world!\nSecond line."
