import pytest
import torch
from src.models.FYNet import (
    FYNetForward,
    FYNetInverse,
    load_FYNetInverse_from_state_dict,
)
from src.models.MFISNet_Fused import (
    MFISNet_Fused,
    load_MFISNet_Fused_from_state_dict,
)
from src.utils.test_utils import (
    check_arrays_close,
)

class Test_FYNetForward:
    def test_0(self: None) -> None:
        """Make sure it initializes and returns the correct shape"""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_M, N_H, 2)
        input_shape = (batch, N_theta, N_rho)

        N_cnn_1d = 4
        c = 7
        w = 5

        # x = FYNetForward(N_cnn_1d, c, w, N_rho, N_H, N_channels_out=1,)
        x = FYNetForward(N_H, N_rho, c, w, N_cnn_1d, N_channels_out=1)

        inputs = torch.randn(input_shape, dtype=torch.complex64)
        print("test: inputs shape: ", inputs.shape)
        y = x(inputs).squeeze(1)

        assert y.shape == output_shape

    def test_1(self: None) -> None:
        """Test with 1 CNN layer to make sure dimensions are still correct."""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_M, N_H, 2)
        input_shape = (batch, N_theta, N_rho)

        N_cnn_1d = 1
        c = 7
        w = 3

        # x = FYNetForward(N_cnn_1d, c, w, N_rho, N_H, N_channels_out=1)
        x = FYNetForward(N_H, N_rho, c, w, N_cnn_1d, N_channels_out=1)

        inputs = torch.randn(input_shape, dtype=torch.complex64)
        print("test: inputs shape: ", inputs.shape)
        y = x(inputs).squeeze(1)

        assert y.shape == output_shape

        assert not torch.any(torch.isnan(y))

    @pytest.mark.skip()
    def test_2(self: None) -> None:
        """Tests that 1 CNN layer is the same as multiplying by a banded
        convolution matrix
        """
        N_batch = 1
        N_M = 1
        N_H = 12

        N_theta = 12
        N_rho = 1
        data = torch.zeros((N_batch, N_theta, N_rho))
        data[0, 0, 0] = 1.0
        data[0, 1, 0] = 3.0
        A = 1.0
        B = 2.0
        N = 12
        on_diag = A * torch.ones(N)
        off_diag = B * torch.ones(N - 1)

        banded_mat = (
            torch.diag(on_diag)
            + torch.diag(off_diag, diagonal=-1)
            + torch.diag(off_diag, diagonal=1)
        )
        banded_mat[0, N - 1] = B
        banded_mat[N - 1, 0] = B

        prod = torch.matmul(banded_mat, data)
        # x = FYNetForward(1, 1, 3, 1, 1, N_channels_out=1)
        
        # x = FYNetForward(N_cnn_1d, c, w, N_rho, N_H, N_channels_out=1)
        x = FYNetForward(1, 1, 1, 3, 1, N_channels_out=1)
        conv_weights = torch.Tensor([[[B, A, B]]]).to(torch.complex64)
        assert conv_weights.shape == x.conv_1d_layers[0].weight.shape

        conv_weights = torch.nn.Parameter(conv_weights)
        # conv_weights = torch.nn.Parameter(conv_weights.view((2, 1, 3)))

        x.conv_1d_layers[0].weight = conv_weights
        x.conv_1d_layers[0].bias = torch.nn.Parameter(
            torch.zeros_like(x.conv_1d_layers[0].bias)
        )
        print("Weight dtype", x.conv_1d_layers[0].weight.dtype)
        print("Bias dtype", x.conv_1d_layers[0].bias.dtype)
        data = data.to(torch.complex64)
        out = x(data).squeeze(1)
        print("Prod shape: ", prod.shape)
        print("Out shape: ", out.shape)
        check_arrays_close(prod.numpy(), out[:, :, :].real.detach().numpy())


class Test_FYNetInverse:
    def test_0(self: None) -> None:
        """Make sure it initializes and returns the correct shape"""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_theta, N_rho)
        input_shape = (batch, N_M, N_H, 2)

        N_cnn_1d = 4
        N_cnn_2d = 4
        c_1d = 5
        c_2d = 13
        w_1d = 5
        w_2d = 3

        x = FYNetInverse(N_H, N_rho, c_1d, c_2d, w_1d, w_2d, N_cnn_1d, N_cnn_2d)

        inputs = torch.randn(input_shape)
        y = x(inputs).squeeze(1)

        assert y.shape == output_shape
        # assert False

    def test_1(self: None) -> None:
        """Make sure it initializes and returns the correct shape with 1 2d conv
        layer
        """

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_theta, N_rho)
        input_shape = (batch, N_M, N_H, 2)

        N_cnn_1d = 4
        N_cnn_2d = 1
        c_1d = 7
        c_2d = 7
        w_1d = 5
        w_2d = 3

        x = FYNetInverse(N_H, N_rho, c_1d, c_2d, w_1d, w_2d, N_cnn_1d, N_cnn_2d)

        inputs = torch.randn(input_shape)
        y = x(inputs)

        assert y.shape == output_shape

    def test_load_from_state_dict_matches_MFISNet_Fused(self: None) -> None:
        """FYNetInverse is a thin, zero-parameter subclass of MFISNet_Fused
        (N_freqs=1); a checkpoint trained as MFISNet_Fused(N_freqs=1) must
        load into FYNetInverse and produce bit-identical outputs, including
        when input/output scaling stats are present in the state dict.
        """
        N_H = 6
        N_rho = 5
        c_1d = 3
        c_2d = 3
        w_1d = 3
        w_2d = 3
        N_cnn_1d = 2
        N_cnn_2d = 2
        batch = 2

        torch.manual_seed(0)
        source_model = MFISNet_Fused(
            N_h=N_H, N_rho=N_rho, N_freqs=1,
            c_1d=c_1d, c_2d=c_2d, w_1d=w_1d, w_2d=w_2d,
            N_cnn_1d=N_cnn_1d, N_cnn_2d=N_cnn_2d, big_init=True,
            train_inputs_mean=torch.tensor(0.1), train_inputs_std=torch.tensor(1.5),
            train_outputs_mean=torch.tensor(0.2), train_outputs_std=torch.tensor(2.0),
        )
        state_dict = source_model.state_dict()

        fused_model  = load_MFISNet_Fused_from_state_dict(state_dict, 1)
        fynet_model  = load_FYNetInverse_from_state_dict(state_dict)

        assert fynet_model.scale_inputs
        assert fynet_model.scale_outputs

        x = torch.randn(batch, 1, N_H, N_H, 2)  # (batch, N_freqs, N_M, N_H, 2)
        x_fynet = x[:, 0]  # (batch, N_M, N_H, 2)

        with torch.no_grad():
            out_fused = fused_model(x)
            out_fynet = fynet_model(x_fynet)

        check_arrays_close(out_fused.numpy(), out_fynet.numpy())


if __name__ == "__main__":
    pytest.main()
