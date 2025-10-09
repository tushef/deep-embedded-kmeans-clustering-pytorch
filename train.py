if __name__ == "__main__":
    from DEKM import DEKM
    from dataset import DatasetWrapper

    dataset = DatasetWrapper('mnist')

    model = DEKM(
        n_clusters=10,
        embedding_size=16,
        batch_size=128,
        pretrain_epochs=100,
        clustering_epochs=150,
        pretrain_learning_rate=0.001,
        clustering_learning_rate=0.0001
    )

    model.fit(dataset)

    predictions = model.predict(dataset.get_full_data().data)