def dataAugmentation(image_flat):
    dataset_size = int(image_flat.shape[0])
    flat_dim = image_flat.shape[1]
    width, height = int(flat_dim ** 0.5), int(flat_dim ** 0.5)
    print(image_flat)

    # Reshape the flattened array
    img_tensor = torch.from_numpy(image_flat).reshape(dataset_size, 1, height, width) / 255.0

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Rotation and translation
        transforms.ToTensor()
    ])

    # Apply the transformations
    img_transformed = torch.stack([transform(image) for image in img_tensor])
    print(img_transformed)
    print(img_transformed.shape)

    # Flatten the transformed NumPy array if needed
    img_transformed_flat = img_transformed.reshape(dataset_size, flat_dim).numpy()
    print(img_transformed_flat)
    print(img_transformed_flat.shape)
    print(np.array_equal(image_flat, img_transformed_flat))

    return img_transformed_flat


def get_input_shape(x):
    """
         x: ndarray
        The data which needs to be interpreted
    """
    # assume square images
    width = int(np.sqrt(x.shape[-1]))
    if width * width == x.shape[-1]:  # gray
        im_shape = [-1, width, width, 1]
    else:  # RGB
        width = int(np.sqrt(x.shape[-1] / 3.0))
        im_shape = [-1, width, width, 3]
    return im_shape


# Simple tensor to image translation
def tensor2img(tensor):
    img = tensor.cpu().data[0]
    if img.shape[0] != 1:
        img = inv_normalize(img)
    img = torch.clamp(img, 0, 1)
    return img


inv_normalize = transforms.Normalize(
    mean=[-0.485 / .229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)
