//
//  TabBarModel.swift
//  ECommerceAppSwiftUI

import SwiftUI

struct TabBarModel: Identifiable {
    
    let id = UUID()
    let name: String
    let imageUrl: String
}

extension TabBarModel {
    
    static func all() -> [TabBarModel] {
        return [
            TabBarModel(name: "Home", imageUrl: "house.fill"),
            TabBarModel(name: "Shop", imageUrl: "cart.fill"),
            TabBarModel(name: "Favorite", imageUrl: "heart.fill"),
            TabBarModel(name: "Profile", imageUrl: "person.fill")
        ]
    }
}
