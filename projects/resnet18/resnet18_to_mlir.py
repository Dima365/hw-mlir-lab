import torch
from torchvision.models.quantization import resnet18, ResNet18_QuantizedWeights


def main():
    torch.backends.quantized.engine = "fbgemm"

    model = resnet18(
        weights=ResNet18_QuantizedWeights.DEFAULT,
        quantize=True,
    ).eval()

    example_input = torch.randn(1, 3, 224, 224)

    output_path = "resnet18_quantized.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
