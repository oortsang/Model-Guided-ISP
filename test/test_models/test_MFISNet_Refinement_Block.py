import pytest
import torch
from src.models.MFISNet_Refinement_Block import KLayer2DCNN


class Test_KLayer2DCNN:
    def test_no_skip_connection(self) -> None:
        """Makes sure the model compiles and runs without error."""
        N_x = 10
        N_y = 10
        c_in = 4
        c_out = 5
        c_feature = 7
        w = 3
        N_cnn = 3
        batch = 5

        model = KLayer2DCNN(
            n_layers=N_cnn,
            n_in_channels=c_in,
            n_out_channels=c_out,
            n_feature_channels=c_feature,
            kernel_size=w,
        )

        x_in = torch.randn((batch, c_in, N_x, N_y))

        x_out = model(x_in)

        assert x_out.shape == (batch, c_out, N_x, N_y)

    def test_skip_connection(self) -> None:
        """Makes sure the model compiles and runs without error."""
        N_x = 10
        N_y = 10
        c_in = 4
        c_out = 5
        c_feature = 7
        w = 3
        N_cnn = 3
        batch = 5

        model = KLayer2DCNN(
            n_layers=N_cnn,
            n_in_channels=c_in,
            n_out_channels=c_out,
            n_feature_channels=c_feature,
            kernel_size=w,
            skip_connection=True,
        )

        x_in = torch.randn((batch, c_in, N_x, N_y))

        x_out = model(x_in)

        assert x_out.shape == (batch, c_out, N_x, N_y)


if __name__ == "__main__":
    pytest.main()
