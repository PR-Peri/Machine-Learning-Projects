//
//  SortByModel.swift
//  ECommerceAppSwiftUI

import Foundation

struct SortByModel: Identifiable {
    let id = UUID()
    let name: String
}

extension SortByModel {
    static func all() -> [SortByModel] {
        return [
            SortByModel(name: "Popular"),
            SortByModel(name: "Newest"),
            SortByModel(name: "Customer review"),
            SortByModel(name: "Price: lowest to high"),
            SortByModel(name: "Price: highest to low")
        ]
    }
}
